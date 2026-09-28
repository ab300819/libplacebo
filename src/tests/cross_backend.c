/*
 * This file is part of libplacebo.
 *
 * libplacebo is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * libplacebo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with libplacebo.  If not, see <http://www.gnu.org/licenses/>.
 */

// Same-host cross-backend differential regression test (Metal vs Vulkan).
//
// This is not an independent oracle. On macOS the Vulkan reference is MoltenVK,
// itself a Metal translation layer, so both paths end up as MSL compiled by the
// same Metal compiler on the same GPU. What the comparison establishes is that
// the two backends agree end to end -- format selection, upload, descriptor
// binding, shader translation, draw, sizing, synchronization, download -- which
// is exactly where a new backend's wiring bugs live. It does not establish that
// the colours are correct (a shared mistake in the common GLSL would agree on
// both sides), nor equivalence against an independent Vulkan driver.
//
// Two separate questions are asked of every scene, because they are not the
// same question:
//
//   gate     Does the output still satisfy the numeric contract? This is the
//            hard pass/fail, and its tolerance depends on what arithmetic the
//            scene performs. Tolerances are non-zero even for plain arithmetic:
//            IEEE-754 pins the rounding of individual operations, not whole
//            expressions, so FMA contraction, reassociation, subnormal flush
//            and sampler-internal precision are all still in play.
//
//   sentinel Is the output still bit-for-bit identical to the other backend?
//            Compared on IEEE-754 bit patterns, not with a zero tolerance:
//            +0.0 and -0.0 are numerically equal but not the same value, and a
//            sentinel that cannot tell them apart cannot claim bit identity.
//            Currently it is, for every scene, but that is a property of this
//            pinned toolchain (Xcode/Metal compiler, MoltenVK, SPIRV-Cross,
//            GPU family), not a guarantee -- the native backend and MoltenVK
//            configure SPIRV-Cross independently. Losing it is reported as
//            drift for a human to look at, not failed, since a compiler update
//            may legitimately change it while the backend stays correct.
//
// L-infinity is the primary metric throughout: PSNR averages over the buffer
// and stays high even when a few pixels are completely wrong, which is the
// shape a stride or addressing bug takes. Non-finite values are rejected up
// front, because every comparison here is blind to NaN.
//
// Peak detection is excluded from these scenes to keep shader precision
// separable from cross-frame state, not because its atomics are unordered --
// they reduce quantized integers, which is order-independent. It needs its own
// test: it is the only path exercising compute, shared memory, SSBOs and
// atomics, none of which this matrix covers.

#include <vulkan/vulkan.h>

#include "utils.h"
#include "image_cmp.h"

#include <libplacebo/metal.h>
#include <libplacebo/vulkan.h>
#include <libplacebo/renderer.h>

#define SRC_W 50
#define SRC_H 50

// Hard-gate tolerance classes. See the header comment for why even the
// arithmetic-only class does not gate on exact equality.
enum cmp_gate {
    GATE_ARITHMETIC,    // basic arithmetic and resampling: max_diff <= 1e-6
    GATE_TRANSCENDENTAL, // pow/exp/log: PSNR >= 80 dB and max_diff <= 1e-3
};

// Resource topology. All the arithmetic-focused scenes share a single-plane
// single-component source, so they exercise the same binding path; YUV420 adds
// the multi-plane one, where plane mapping and chroma sizing can go wrong
// independently of any shader maths.
enum cmp_topology {
    TOPO_SINGLE_PLANE,
    TOPO_YUV420,
};

struct scene {
    const char *name;
    enum cmp_gate gate;
    enum cmp_topology topology;
    const struct pl_color_space *src_csp;
    int scale_num, scale_den; // target size = source size * num / den
    const struct pl_filter_config *upscaler;
};

static const struct scene scenes[] = {
    {
        // Same colorspace both ends, 1:1, float target: no tone mapping, no
        // resampling, no dithering.
        .name = "passthrough",
        .gate = GATE_ARITHMETIC,
        .src_csp = &pl_color_space_srgb,
        .scale_num = 1, .scale_den = 1,
    }, {
        .name = "upscale-2x-lanczos",
        .gate = GATE_ARITHMETIC,
        .src_csp = &pl_color_space_srgb,
        .scale_num = 2, .scale_den = 1,
        .upscaler = &pl_filter_lanczos,
    }, {
        // jinc-jinc: a polar filter, so more taps and more accumulation
        .name = "upscale-2x-ewa-lanczos",
        .gate = GATE_ARITHMETIC,
        .src_csp = &pl_color_space_srgb,
        .scale_num = 2, .scale_den = 1,
        .upscaler = &pl_filter_ewa_lanczos,
    }, {
        .name = "downscale-half-lanczos",
        .gate = GATE_ARITHMETIC,
        .src_csp = &pl_color_space_srgb,
        .scale_num = 1, .scale_den = 2,
        .upscaler = &pl_filter_lanczos,
    }, {
        // PQ BT.2020 -> sRGB exercises the tone and gamut mapping paths, which
        // is where pow/exp/log live.
        .name = "tonemap-hdr10-to-srgb",
        .gate = GATE_TRANSCENDENTAL,
        .src_csp = &pl_color_space_hdr10,
        .scale_num = 1, .scale_den = 1,
    }, {
        // Three planes with half-size chroma: the YCbCr matrix is plain
        // arithmetic, but reaching it requires correct plane mapping, chroma
        // shift and per-plane scaling across three separate bindings.
        .name = "yuv420-3plane-to-srgb",
        .gate = GATE_ARITHMETIC,
        .topology = TOPO_YUV420,
        .src_csp = &pl_color_space_bt709,
        .scale_num = 1, .scale_den = 1,
    },
};

// What one backend produced for one scene.
struct render_out {
    float *data;          // caller frees
    int count;            // number of floats
    int num_comps;        // components per pixel in `data`
    const char *src_fmt;  // format chosen for the source plane(s)
    const char *fbo_fmt;  // format chosen for the render target
};

static void render_out_free(struct render_out *o)
{
    free(o->data);
    o->data = NULL;
}

// Render one scene on a given GPU and download the result. Reports the chosen
// formats as well: both backends pick these independently from their own format
// tables, and comparing outputs produced at different precisions would silently
// attribute an input quantization difference to the shader.
static bool render_scene(pl_gpu gpu, const struct scene *s,
                         struct render_out *out)
{
    *out = (struct render_out) { .src_fmt = "(none)", .fbo_fmt = "(none)" };

    pl_fmt src_fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 1, 16, 32,
                                 PL_FMT_CAP_SAMPLEABLE);
    pl_fmt fbo_fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 16, 32,
                                 PL_FMT_CAP_RENDERABLE);
    if (!src_fmt || !fbo_fmt)
        return false;
    if (!(fbo_fmt->caps & PL_FMT_CAP_HOST_READABLE))
        return false;

    out->src_fmt = src_fmt->name;
    out->fbo_fmt = fbo_fmt->name;

    int dst_w = SRC_W * s->scale_num / s->scale_den;
    int dst_h = SRC_H * s->scale_num / s->scale_den;
    int num_comps = fbo_fmt->num_components;
    int pixel_count = dst_w * dst_h * num_comps;
    out->num_comps = num_comps;

    bool yuv = s->topology == TOPO_YUV420;
    int cw = (SRC_W + 1) / 2, ch = (SRC_H + 1) / 2;

    // Deterministic source data, reseeded per call so both backends see the
    // same pixels. Each plane gets its own seed: identical chroma planes would
    // make a U/V mix-up invisible.
    float y_data[SRC_W * SRC_H];
    float u_data[((SRC_W + 1) / 2) * ((SRC_H + 1) / 2)];
    float v_data[((SRC_W + 1) / 2) * ((SRC_H + 1) / 2)];
    srand(42);
    for (int i = 0; i < SRC_W * SRC_H; i++)
        y_data[i] = (float) rand() / (float) RAND_MAX;
    srand(1337);
    for (int i = 0; i < cw * ch; i++)
        u_data[i] = (float) rand() / (float) RAND_MAX;
    srand(24601);
    for (int i = 0; i < cw * ch; i++)
        v_data[i] = (float) rand() / (float) RAND_MAX;

    bool ok = false;
    pl_tex planes[3] = {0};
    pl_tex fbo = NULL;
    pl_renderer rr = NULL;

    planes[0] = pl_tex_create(gpu, pl_tex_params(
        .w = SRC_W, .h = SRC_H, .format = src_fmt,
        .sampleable = true, .host_writable = true, .initial_data = y_data,
    ));
    if (yuv) {
        planes[1] = pl_tex_create(gpu, pl_tex_params(
            .w = cw, .h = ch, .format = src_fmt,
            .sampleable = true, .host_writable = true, .initial_data = u_data,
        ));
        planes[2] = pl_tex_create(gpu, pl_tex_params(
            .w = cw, .h = ch, .format = src_fmt,
            .sampleable = true, .host_writable = true, .initial_data = v_data,
        ));
        if (!planes[1] || !planes[2])
            goto cleanup;
    }

    fbo = pl_tex_create(gpu, pl_tex_params(
        .w = dst_w, .h = dst_h, .format = fbo_fmt,
        .renderable = true, .host_readable = true,
        .storable = fbo_fmt->caps & PL_FMT_CAP_STORABLE,
    ));
    rr = pl_renderer_create(gpu->log, gpu);
    if (!planes[0] || !fbo || !rr)
        goto cleanup;

    struct pl_frame image = {
        .num_planes = 1,
        .planes = {{
            .texture = planes[0],
            .components = 1,
            .component_mapping = {0},
        }},
        .repr = {
            .sys = PL_COLOR_SYSTEM_BT_709,
            .levels = PL_COLOR_LEVELS_FULL,
        },
        .color = *s->src_csp,
    };

    if (yuv) {
        float sx, sy;
        pl_chroma_location_offset(PL_CHROMA_LEFT, &sx, &sy);
        image.num_planes = 3;
        image.planes[1] = (struct pl_plane) {
            .texture = planes[1],
            .components = 1,
            .component_mapping = {1},
            .shift_x = sx, .shift_y = sy,
        };
        image.planes[2] = (struct pl_plane) {
            .texture = planes[2],
            .components = 1,
            .component_mapping = {2},
            .shift_x = sx, .shift_y = sy,
        };
        image.repr = pl_color_repr_hdtv;
    }

    struct pl_frame target = {
        .num_planes = 1,
        .planes = {{
            .texture = fbo,
            .components = num_comps,
            .component_mapping = {0, 1, 2, 3},
        }},
        .repr = {
            .sys = PL_COLOR_SYSTEM_RGB,
            .levels = PL_COLOR_LEVELS_FULL,
            // .bits left zeroed on purpose: the FBO is float, and setting
            // color_depth without sample_depth makes pl_color_repr_normalize()
            // scale the output by (2^32-1)/255.
        },
        .color = pl_color_space_srgb,
    };

    struct pl_render_params params = pl_render_default_params;
    if (s->upscaler) {
        params.upscaler = s->upscaler;
        params.downscaler = s->upscaler;
    }
    // Stateful across frames, and the only compute/SSBO/atomics path here, so
    // it needs a differential test of its own rather than riding along in a
    // comparison aimed at shader precision.
    params.peak_detect_params = NULL;

    if (!pl_render_image(rr, &image, &target, &params))
        goto cleanup;

    out->data = malloc(pixel_count * sizeof(float));
    if (!out->data)
        goto cleanup;

    if (!pl_tex_download(gpu, pl_tex_transfer_params(
            .tex = fbo,
            .ptr = out->data,
        )))
    {
        free(out->data);
        out->data = NULL;
        goto cleanup;
    }

    out->count = pixel_count;
    ok = true;

cleanup:
    pl_renderer_destroy(&rr);
    for (int i = 0; i < 3; i++)
        pl_tex_destroy(gpu, &planes[i]);
    pl_tex_destroy(gpu, &fbo);
    return ok;
}

// Reject a frame that carries no image.
//
// Per channel, not pooled across channels: a flat fill still varies *between*
// channels, so a pooled range accepts it. An all-black frame is {0,0,0,1} and a
// flat red one is {1,0,0,1}; both span 1.0 once alpha or the other channels are
// mixed in, which is exactly the case this is meant to reject. What matters is
// whether any single colour channel varies from pixel to pixel.
static bool has_image_content(const char *who, const struct render_out *o)
{
    if (o->num_comps <= 0 || o->count % o->num_comps != 0) {
        printf("    %s produced %d values in %d-component pixels\n",
               who, o->count, o->num_comps);
        return false;
    }

    int colour_comps = PL_MIN(o->num_comps, 3);
    float lo[3], hi[3];
    for (int c = 0; c < colour_comps; c++) {
        lo[c] = INFINITY;
        hi[c] = -INFINITY;
    }

    for (int i = 0; i < o->count; i += o->num_comps) {
        for (int c = 0; c < colour_comps; c++) {
            lo[c] = fminf(lo[c], o->data[i + c]);
            hi[c] = fmaxf(hi[c], o->data[i + c]);
        }
    }

    for (int c = 0; c < colour_comps; c++) {
        if (hi[c] - lo[c] > 1e-6f)
            return true;
    }

    printf("    %s produced a spatially flat frame; per-channel ranges:", who);
    for (int c = 0; c < colour_comps; c++)
        printf(" [%f, %f]", lo[c], hi[c]);
    printf("\n");
    return false;
}

// Apply both layers and report. Returns false only when the hard gate fails,
// so a single run shows every scene's numbers instead of stopping at the first
// divergence, and drift is surfaced without blocking.
static bool check_scene(const struct scene *s, const float *a, const float *b,
                        int count, int *out_drift)
{
    // Must come first: max_diff and the tolerance check are both blind to NaN
    int nonfinite = img_cmp_count_nonfinite(a, b, count);
    if (nonfinite) {
        printf("  %-24s %d/%d non-finite values  [FAIL: output not finite]\n",
               s->name, nonfinite, count);
        return false;
    }

    double psnr = img_cmp_psnr(a, b, count);
    double mse = img_cmp_mse(a, b, count);
    float max_diff = img_cmp_max_diff(a, b, count);

    float tol;
    bool ok;
    const char *criterion;

    switch (s->gate) {
    case GATE_ARITHMETIC:
        tol = 1e-6f;
        criterion = "max_diff <= 1e-6";
        ok = max_diff <= tol;
        break;
    case GATE_TRANSCENDENTAL:
        tol = 1e-3f;
        criterion = "PSNR >= 80 dB && max_diff <= 1e-3";
        ok = psnr >= 80.0 && max_diff <= tol;
        break;
    default:
        return false;
    }

    int over_gate = img_cmp_count_above(a, b, count, tol);
    // Sentinel: real bit comparison, so +-0.0 counts as a difference
    int differing = img_cmp_count_bitdiff(a, b, count);
    bool drift = differing > 0;

    printf("  %-24s PSNR %8.2f dB  MSE %9.3e  max_diff %9.3e  "
           "over-gate %d/%d  differing %d/%d  [%s] %s%s\n",
           s->name, psnr, mse, max_diff, over_gate, count, differing, count,
           criterion, ok ? "PASS" : "FAIL", drift ? " (DRIFT)" : "");

    if (drift)
        (*out_drift)++;
    return ok;
}

// The differential comparison above cannot see a mistake made identically by
// both backends -- swapping the U and V mapping on both keeps them in perfect
// agreement. So assert the actual colour direction once per backend: feed
// neutral Cb with a raised Cr and the result must lean red, which a swapped
// mapping turns blue.
static bool check_chroma_direction(const char *who, pl_gpu gpu)
{
    enum { W = 8, H = 8, CW = 4, CH = 4, COMPS = 4 };

    pl_fmt src_fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 1, 16, 32,
                                 PL_FMT_CAP_SAMPLEABLE);
    pl_fmt fbo_fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 16, 32,
                                 PL_FMT_CAP_RENDERABLE);
    if (!src_fmt || !fbo_fmt || !(fbo_fmt->caps & PL_FMT_CAP_HOST_READABLE))
        return true; // nothing to assert on this GPU

    float y[W * H], cb[CW * CH], cr[CW * CH];
    for (int i = 0; i < W * H; i++)
        y[i] = 0.5f;
    for (int i = 0; i < CW * CH; i++) {
        cb[i] = 128.0f / 255.0f; // neutral
        cr[i] = 0.8f;            // toward red
    }

    bool ok = false;
    pl_tex ty = NULL, tcb = NULL, tcr = NULL, fbo = NULL;
    pl_renderer rr = NULL;
    float *px = NULL;

    ty = pl_tex_create(gpu, pl_tex_params(.w = W, .h = H, .format = src_fmt,
        .sampleable = true, .host_writable = true, .initial_data = y));
    tcb = pl_tex_create(gpu, pl_tex_params(.w = CW, .h = CH, .format = src_fmt,
        .sampleable = true, .host_writable = true, .initial_data = cb));
    tcr = pl_tex_create(gpu, pl_tex_params(.w = CW, .h = CH, .format = src_fmt,
        .sampleable = true, .host_writable = true, .initial_data = cr));
    fbo = pl_tex_create(gpu, pl_tex_params(.w = W, .h = H, .format = fbo_fmt,
        .renderable = true, .host_readable = true));
    rr = pl_renderer_create(gpu->log, gpu);
    if (!ty || !tcb || !tcr || !fbo || !rr)
        goto cleanup;

    float sx, sy;
    pl_chroma_location_offset(PL_CHROMA_LEFT, &sx, &sy);
    struct pl_frame image = {
        .num_planes = 3,
        .planes = {
            { .texture = ty,  .components = 1, .component_mapping = {0} },
            { .texture = tcb, .components = 1, .component_mapping = {1},
              .shift_x = sx, .shift_y = sy },
            { .texture = tcr, .components = 1, .component_mapping = {2},
              .shift_x = sx, .shift_y = sy },
        },
        .repr = pl_color_repr_hdtv,
        .color = pl_color_space_bt709,
    };
    struct pl_frame target = {
        .num_planes = 1,
        .planes = {{ .texture = fbo, .components = COMPS,
                     .component_mapping = {0, 1, 2, 3} }},
        .repr = { .sys = PL_COLOR_SYSTEM_RGB, .levels = PL_COLOR_LEVELS_FULL },
        .color = pl_color_space_srgb,
    };

    if (!pl_render_image(rr, &image, &target, &pl_render_default_params))
        goto cleanup;

    px = malloc((size_t) W * H * COMPS * sizeof(*px));
    if (!px || !pl_tex_download(gpu, pl_tex_transfer_params(.tex = fbo, .ptr = px)))
        goto cleanup;

    double r = 0, b = 0;
    for (int i = 0; i < W * H * COMPS; i += COMPS) {
        r += px[i];
        b += px[i + 2];
    }
    printf("  %-24s mean R %.4f  mean B %.4f  %s\n",
           who, r / (W * H), b / (W * H), r > b ? "red-leaning" : "NOT red");
    ok = r > b;

cleanup:
    free(px);
    pl_renderer_destroy(&rr);
    pl_tex_destroy(gpu, &ty);
    pl_tex_destroy(gpu, &tcb);
    pl_tex_destroy(gpu, &tcr);
    pl_tex_destroy(gpu, &fbo);
    return ok;
}

int main(void)
{
    pl_log log = pl_test_logger();

    pl_metal metal = pl_metal_create(log, NULL);
    if (!metal) {
        printf("Metal not available, skipping cross-backend test\n");
        pl_log_destroy(&log);
        return SKIP;
    }

    pl_vk_inst vk_inst = pl_vk_inst_create(log, pl_vk_inst_params(
        .get_proc_addr = vkGetInstanceProcAddr,
    ));

    pl_vulkan vk = NULL;
    if (vk_inst) {
        VkPhysicalDevice phys_dev = pl_vulkan_choose_device(log,
            pl_vulkan_device_params(
                .instance = vk_inst->instance,
                .get_proc_addr = vk_inst->get_proc_addr,
                .allow_software = true,
            ));
        if (phys_dev) {
            vk = pl_vulkan_create(log, pl_vulkan_params(
                .instance = vk_inst->instance,
                .get_proc_addr = vk_inst->get_proc_addr,
                .device = phys_dev,
            ));
        }
    }

    if (!vk) {
        printf("Vulkan not available, skipping cross-backend comparison\n");
        pl_metal_destroy(&metal);
        if (vk_inst)
            pl_vk_inst_destroy(&vk_inst);
        pl_log_destroy(&log);
        return SKIP;
    }

    printf("cross_backend_tests: Metal vs Vulkan (same-host differential)\n");

    int failures = 0, drifted = 0;
    for (int i = 0; i < (int) PL_ARRAY_SIZE(scenes); i++) {
        const struct scene *s = &scenes[i];
        struct render_out m = {0}, v = {0};

        REQUIRE(render_scene(metal->gpu, s, &m));
        REQUIRE(render_scene(vk->gpu, s, &v));
        REQUIRE_CMP(m.count, ==, v.count, "d");
        REQUIRE_CMP(m.count, >, 0, "d");

        // Both backends must have rendered at the same precision, on both the
        // input and the output side, or a format difference would masquerade
        // as a backend difference.
        REQUIRE_STREQ(m.src_fmt, v.src_fmt);
        REQUIRE_STREQ(m.fbo_fmt, v.fbo_fmt);

        // Two backends that both rendered nothing would compare equal, so
        // check each side independently before comparing them.
        REQUIRE(has_image_content("metal", &m));
        REQUIRE(has_image_content("vulkan", &v));

        printf("  %-24s src=%s fbo=%s\n", s->name, m.src_fmt, m.fbo_fmt);
        if (!check_scene(s, m.data, v.data, m.count, &drifted))
            failures++;

        render_out_free(&m);
        render_out_free(&v);
    }

    printf("chroma direction probe (raised Cr must lean red):\n");
    REQUIRE(check_chroma_direction("metal", metal->gpu));
    REQUIRE(check_chroma_direction("vulkan", vk->gpu));

    REQUIRE(!pl_gpu_is_failed(metal->gpu));
    REQUIRE(!pl_gpu_is_failed(vk->gpu));

    pl_vulkan_destroy(&vk);
    pl_vk_inst_destroy(&vk_inst);
    pl_metal_destroy(&metal);
    pl_log_destroy(&log);

    // Drift is reported, not failed: the sentinel tracks this pinned toolchain,
    // and a compiler or MoltenVK update may legitimately move it. The driver
    // and OS versions that produced this run are in the log above.
    if (drifted) {
        printf("NOTE: %d/%d scenes are no longer bit-identical between "
               "backends while still inside their gate. This is drift against "
               "the current toolchain, not a rendering error -- review the "
               "per-scene max_diff above and re-baseline if the toolchain "
               "changed.\n", drifted, (int) PL_ARRAY_SIZE(scenes));
    }

    if (failures) {
        fprintf(stderr, "%d/%d scenes exceeded their gate tolerance\n",
                failures, (int) PL_ARRAY_SIZE(scenes));
        return 1;
    }
    return 0;
}
