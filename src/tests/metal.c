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

#include "gpu_tests.h"
#include "image_cmp.h"
#include "shaders.h"

#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CGColor.h>
#include <CoreGraphics/CGColorSpace.h>
#include <CoreGraphics/CGGeometry.h>
#include <IOKit/IOReturn.h>
#include <IOSurface/IOSurfaceRef.h>
#include <objc/message.h>
#include <objc/runtime.h>

// Public ObjC runtime entry points, not declared in any SDK header
void *objc_autoreleasePoolPush(void);
void objc_autoreleasePoolPop(void *pool);

#include <libplacebo/dispatch.h>
#include <libplacebo/metal.h>
#include <libplacebo/renderer.h>
#include <libplacebo/shaders/sampling.h>

static void metal_interop_tests(pl_gpu gpu)
{
    // Test MTL_TEX export/import roundtrip
    if (!(gpu->export_caps.tex & PL_HANDLE_MTL_TEX) ||
        !(gpu->import_caps.tex & PL_HANDLE_MTL_TEX))
    {
        printf("pl_metal_interop_tests: MTL_TEX not in %s caps"
               "... skipping this test\n",
               (gpu->export_caps.tex & PL_HANDLE_MTL_TEX) ? "import" : "export");
        return;
    }

    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_UNORM, 4, 0, 0, PL_FMT_CAP_BLITTABLE);
    if (!fmt) {
        printf("pl_metal_interop_tests: no blittable 4-component UNORM format"
               "... skipping this test\n");
        return;
    }

    printf("pl_metal_interop_tests:\n");

    // Export a texture and verify the handle
    printf("- testing MTL_TEX export\n");
    pl_tex export_tex = pl_tex_create(gpu, pl_tex_params(
        .w = 32,
        .h = 32,
        .format = fmt,
        .export_handle = PL_HANDLE_MTL_TEX,
    ));
    REQUIRE(export_tex);
    REQUIRE(export_tex->shared_mem.handle.handle);

    // Import the exported texture
    printf("- testing MTL_TEX import from export\n");
    pl_tex import_tex = pl_tex_create(gpu, pl_tex_params(
        .w = export_tex->params.w,
        .h = export_tex->params.h,
        .format = fmt,
        .import_handle = PL_HANDLE_MTL_TEX,
        .shared_mem = export_tex->shared_mem,
    ));
    REQUIRE(import_tex);

    pl_tex_destroy(gpu, &import_tex);
    pl_tex_destroy(gpu, &export_tex);

    // Test pl_metal_wrap roundtrip with data
    printf("- testing pl_metal_wrap with data verification\n");
    pl_tex src = pl_tex_create(gpu, pl_tex_params(
        .w = 4,
        .h = 4,
        .format = fmt,
        .export_handle = PL_HANDLE_MTL_TEX,
        .host_writable = true,
        .host_readable = true,
        .sampleable = true,
        .renderable = true,
        .blit_src = true,
        .blit_dst = true,
    ));
    REQUIRE(src);

    // Upload test pattern
    uint8_t data[4 * 4 * 4]; // 4x4 RGBA8
    for (int i = 0; i < 4 * 4 * 4; i++)
        data[i] = (uint8_t) i;

    REQUIRE(pl_tex_upload(gpu, pl_tex_transfer_params(
        .tex = src,
        .ptr = data,
    )));

    // Wrap the same MTL::Texture via pl_metal_wrap
    pl_tex wrapped = pl_metal_wrap(gpu, pl_metal_wrap_params(
        .texture = src->shared_mem.handle.handle,
        .w = 4,
        .h = 4,
        .format = fmt,
    ));
    REQUIRE(wrapped);

    // Download from the wrapped texture and compare
    uint8_t readback[4 * 4 * 4] = {0};
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(
        .tex = wrapped,
        .ptr = readback,
    )));
    REQUIRE_MEMEQ(data, readback, sizeof(data));

    pl_tex_destroy(gpu, &wrapped);
    pl_tex_destroy(gpu, &src);

    REQUIRE(!pl_gpu_is_failed(gpu));
}

// Build a CFDictionary describing an IOSurface, or NULL on failure. The caller
// owns the result.
static CFDictionaryRef iosurface_props(int w, int h, int bpe, uint32_t fourcc)
{
    CFMutableDictionaryRef props =
        CFDictionaryCreateMutable(kCFAllocatorDefault, 4,
                                  &kCFTypeDictionaryKeyCallBacks,
                                  &kCFTypeDictionaryValueCallBacks);
    if (!props)
        return NULL;

    const struct { const CFStringRef key; int value; } ints[] = {
        { kIOSurfaceWidth,           w    },
        { kIOSurfaceHeight,          h    },
        { kIOSurfaceBytesPerElement, bpe  },
        { kIOSurfacePixelFormat,     (int) fourcc },
    };

    for (int i = 0; i < (int) PL_ARRAY_SIZE(ints); i++) {
        CFNumberRef num = CFNumberCreate(kCFAllocatorDefault, kCFNumberIntType,
                                         &ints[i].value);
        if (!num) {
            CFRelease(props);
            return NULL;
        }
        CFDictionarySetValue(props, ints[i].key, num);
        CFRelease(num);
    }

    return props;
}

static void metal_iosurface_tests(pl_gpu gpu)
{
    if (!(gpu->import_caps.tex & PL_HANDLE_IOSURFACE)) {
        printf("pl_metal_iosurface_tests: IOSURFACE not in import caps"
               "... skipping this test\n");
        return;
    }

    // 'BGRA' is the IOSurface format VideoToolbox emits, and pairs with the
    // BGRA8Unorm Metal pixel format that pl_fmt "bgra8" maps to.
    pl_fmt fmt = pl_find_named_fmt(gpu, "bgra8");
    if (!fmt || !(fmt->caps & PL_FMT_CAP_HOST_READABLE)) {
        printf("pl_metal_iosurface_tests: no host-readable bgra8 format"
               "... skipping this test\n");
        return;
    }

    printf("pl_metal_iosurface_tests:\n");

    enum { W = 8, H = 8, BPE = 4, SIZE = W * H * BPE };

    CFDictionaryRef props = iosurface_props(W, H, BPE, 'BGRA');
    REQUIRE(props);
    IOSurfaceRef surf = IOSurfaceCreate(props);
    CFRelease(props);
    if (!surf) {
        printf("- IOSurfaceCreate failed... skipping this test\n");
        return;
    }

    // Write a known pattern through the CPU mapping. IOSurface rows may be
    // padded, so honour the reported stride rather than assuming W * BPE.
    printf("- writing test pattern via IOSurfaceLock\n");
    uint8_t expected[SIZE];
    for (int i = 0; i < SIZE; i++)
        expected[i] = (uint8_t) (i * 7 + 1);

    REQUIRE(IOSurfaceLock(surf, 0, NULL) == kIOReturnSuccess);
    uint8_t *base = IOSurfaceGetBaseAddress(surf);
    REQUIRE(base);
    size_t stride = IOSurfaceGetBytesPerRow(surf);
    REQUIRE_CMP(stride, >=, (size_t) W * BPE, "zu");
    for (int y = 0; y < H; y++)
        memcpy(base + y * stride, expected + y * W * BPE, W * BPE);
    REQUIRE(IOSurfaceUnlock(surf, 0, NULL) == kIOReturnSuccess);

    // Import it zero-copy and read it back through the GPU
    printf("- importing as PL_HANDLE_IOSURFACE\n");
    pl_tex tex = pl_tex_create(gpu, pl_tex_params(
        .w = W,
        .h = H,
        .format = fmt,
        .import_handle = PL_HANDLE_IOSURFACE,
        .shared_mem = { .handle = { .handle = surf } },
        .sampleable = true,
        .host_readable = true,
    ));
    REQUIRE(tex);
    REQUIRE(tex->shared_mem.handle.handle == surf);

    printf("- verifying imported contents\n");
    uint8_t readback[SIZE] = {0};
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(
        .tex = tex,
        .ptr = readback,
    )));
    REQUIRE_MEMEQ(expected, readback, SIZE);

    pl_tex_destroy(gpu, &tex);
    CFRelease(surf);

    REQUIRE(!pl_gpu_is_failed(gpu));
}

static void metal_render_regression(pl_gpu gpu)
{
    // Find a float RGBA format for the FBO. BLITTABLE is required because the
    // determinism check clears the FBO between passes via pl_tex_clear_ex().
    pl_fmt fbo_fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 16, 32,
                                 PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_BLITTABLE);
    pl_fmt src_fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 1, 16, 32,
                                 PL_FMT_CAP_SAMPLEABLE);
    if (!fbo_fmt || !src_fmt || !(fbo_fmt->caps & PL_FMT_CAP_HOST_READABLE)) {
        printf("metal_render_regression: no renderable+blittable+host-readable "
               "float format (fbo=%s src=%s)... skipping this test\n",
               fbo_fmt ? fbo_fmt->name : "none",
               src_fmt ? src_fmt->name : "none");
        return;
    }

    printf("metal_render_regression: fbo=%s src=%s\n", fbo_fmt->name,
           src_fmt->name);

    enum { W = 50, H = 50 };
    int num_comps = fbo_fmt->num_components;
    int pixel_count = W * H * num_comps;

    // Create deterministic source data (seeded RNG)
    srand(12345);
    float src_data[W * H];
    for (int i = 0; i < W * H; i++)
        src_data[i] = (float) rand() / (float) RAND_MAX;

    pl_tex src_tex = pl_tex_create(gpu, pl_tex_params(
        .w = W,
        .h = H,
        .format = src_fmt,
        .sampleable = true,
        .host_writable = true,
        .initial_data = src_data,
    ));

    pl_tex fbo = pl_tex_create(gpu, pl_tex_params(
        .w = W,
        .h = H,
        .format = fbo_fmt,
        .renderable = true,
        .host_readable = true,
        .blit_dst = true,
        .storable = fbo_fmt->caps & PL_FMT_CAP_STORABLE,
    ));

    float *output1 = NULL, *output2 = NULL;
    pl_renderer rr = pl_renderer_create(gpu->log, gpu);
    if (!src_tex || !fbo || !rr)
        goto error;

    struct pl_plane src_plane = {
        .texture = src_tex,
        .components = 1,
        .component_mapping = {0},
    };

    struct pl_frame image = {
        .num_planes = 1,
        .planes = { src_plane },
        .repr = {
            .sys = PL_COLOR_SYSTEM_BT_709,
            .levels = PL_COLOR_LEVELS_FULL,
        },
        .color = pl_color_space_srgb,
    };

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

    // First render
    printf("- testing deterministic rendering (pass 1)\n");
    REQUIRE(pl_render_image(rr, &image, &target, NULL));
    REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);

    output1 = malloc(pixel_count * sizeof(float));
    REQUIRE(output1);
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(
        .tex = fbo,
        .ptr = output1,
    )));

    // Second render with same params — should be identical
    printf("- testing deterministic rendering (pass 2)\n");
    pl_tex_clear_ex(gpu, fbo, (union pl_clear_color){0});
    REQUIRE(pl_render_image(rr, &image, &target, NULL));
    REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);

    output2 = malloc(pixel_count * sizeof(float));
    REQUIRE(output2);
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(
        .tex = fbo,
        .ptr = output2,
    )));

    double psnr = img_cmp_psnr(output1, output2, pixel_count);
    float max_diff = img_cmp_max_diff(output1, output2, pixel_count);
    printf("  determinism PSNR: %f dB, max_diff: %f\n", psnr, max_diff);
    REQUIRE(isinf(psnr) || psnr >= 60.0); // should be identical or near-identical

    // Stress test: render many times with varying params
    printf("- stress testing (%d renders with varying params)\n", 20);
    struct pl_render_params params = pl_render_default_params;
    for (int i = 0; i < 20; i++) {
        params.force_dither = (i % 2 == 0);
        struct pl_deband_params deband = pl_deband_default_params;
        deband.iterations = i % 4;
        params.deband_params = &deband;
        REQUIRE(pl_render_image(rr, &image, &target, &params));
        pl_gpu_flush(gpu);
        REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
    }

    REQUIRE(!pl_gpu_is_failed(gpu));
    printf("  stress test passed\n");

error:
    free(output1);
    free(output2);
    pl_renderer_destroy(&rr);
    pl_tex_destroy(gpu, &src_tex);
    pl_tex_destroy(gpu, &fbo);
}

static void metal_download_stride_test(pl_gpu gpu)
{
    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_UNORM, 4, 8, 16,
                             PL_FMT_CAP_HOST_READABLE);
    if (!fmt)
        return;

    enum { W = 11, H = 7 };
    size_t row_size = (size_t) W * fmt->texel_size;
    size_t pitch_align = gpu->limits.align_tex_xfer_pitch;
    if (pitch_align < (size_t) fmt->texel_align)
        pitch_align = fmt->texel_align;
    if (!pitch_align)
        pitch_align = 1;

    size_t row_pitch = PL_ALIGN2(row_size + pitch_align, pitch_align);
    size_t buf_size = (size_t) H * row_pitch;

    uint8_t *src = malloc((size_t) W * H * fmt->texel_size);
    uint8_t *dst = malloc(buf_size);
    if (!src || !dst) {
        free(src);
        free(dst);
        return;
    }

    for (size_t i = 0; i < (size_t) W * H * fmt->texel_size; i++)
        src[i] = (uint8_t) (i * 13 + 7);

    memset(dst, 0xCD, buf_size);

    pl_tex tex = pl_tex_create(gpu, pl_tex_params(
        .w = W,
        .h = H,
        .format = fmt,
        .sampleable = true,
        .host_writable = true,
        .host_readable = true,
    ));
    REQUIRE(tex);

    REQUIRE(pl_tex_upload(gpu, pl_tex_transfer_params(
        .tex = tex,
        .ptr = src,
    )));
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(
        .tex = tex,
        .ptr = dst,
        .row_pitch = row_pitch,
    )));

    for (int y = 0; y < H; y++) {
        const uint8_t *row_src = src + (size_t) y * row_size;
        const uint8_t *row_dst = dst + (size_t) y * row_pitch;
        REQUIRE_MEMEQ(row_src, row_dst, row_size);
        for (size_t x = row_size; x < row_pitch; x++)
            REQUIRE_CMP(row_dst[x], ==, 0xCD, "u");
    }

    pl_tex_destroy(gpu, &tex);
    free(src);
    free(dst);
}

// Objective-C helpers. The swapchain takes a CAMetalLayer, and a layer that is
// not attached to a window reports a zero drawableSize (it derives one from its
// bounds), so the test has to set the size explicitly.
static id make_metal_layer(int w, int h)
{
    Class cls = objc_getClass("CAMetalLayer");
    if (!cls)
        return NULL;

    id layer = ((id (*)(Class, SEL)) objc_msgSend)(cls, sel_registerName("layer"));
    if (!layer)
        return NULL;
    layer = ((id (*)(id, SEL)) objc_msgSend)(layer, sel_registerName("retain"));

    ((void (*)(id, SEL, CGSize)) objc_msgSend)(
        layer, sel_registerName("setDrawableSize:"), CGSizeMake(w, h));
    return layer;
}

static CGSize layer_drawable_size(id layer)
{
    return ((CGSize (*)(id, SEL)) objc_msgSend)(
        layer, sel_registerName("drawableSize"));
}

// Read back the swapchain framebuffer. The backbuffer is deliberately not
// host_readable (that would allocate a shadow the size of the screen every
// frame), so blit into a readable staging texture instead. pl_tex_blit()
// requires both formats to have the same internal size, so reuse the
// framebuffer's own format rather than picking a float one.
static bool read_back_fbo(pl_gpu gpu, pl_tex fbo, uint8_t *out)
{
    if (!fbo->params.blit_src || !(fbo->params.format->caps & PL_FMT_CAP_HOST_READABLE))
        return false;

    pl_tex staging = pl_tex_create(gpu, pl_tex_params(
        .w = fbo->params.w,
        .h = fbo->params.h,
        .format = fbo->params.format,
        .blit_dst = true,
        .host_readable = true,
    ));
    if (!staging)
        return false;

    pl_tex_blit(gpu, pl_tex_blit_params(.dst = staging, .src = fbo));
    bool ok = pl_tex_download(gpu, pl_tex_transfer_params(
        .tex = staging,
        .ptr = out,
    ));
    pl_tex_destroy(gpu, &staging);
    return ok;
}

static void metal_swapchain_tests(pl_metal metal)
{
    pl_gpu gpu = metal->gpu;
    enum { W = 64, H = 48 };

    id layer = make_metal_layer(W, H);
    if (!layer) {
        printf("pl_metal_swapchain_tests: CAMetalLayer unavailable"
               "... skipping this test\n");
        return;
    }

    pl_swapchain sw = pl_metal_create_swapchain(metal,
        pl_metal_swapchain_params(.layer = (void *) layer));
    if (!sw) {
        printf("pl_metal_swapchain_tests: swapchain creation failed"
               "... skipping this test\n");
        ((void (*)(id, SEL)) objc_msgSend)(layer, sel_registerName("release"));
        return;
    }

    printf("pl_metal_swapchain_tests:\n");

    // pl_swapchain_resize() must honour a non-zero request and report back what
    // it actually got; 0 means "just tell me the current size".
    printf("- testing resize\n");
    int w = 0, h = 0;
    REQUIRE(pl_swapchain_resize(sw, &w, &h));
    REQUIRE_CMP(w, ==, (int) W, "d");
    REQUIRE_CMP(h, ==, (int) H, "d");

    w = W / 2;
    h = H / 2;
    REQUIRE(pl_swapchain_resize(sw, &w, &h));
    REQUIRE_CMP(w, ==, (int) W / 2, "d");
    REQUIRE_CMP(h, ==, (int) H / 2, "d");
    REQUIRE_CMP((int) layer_drawable_size(layer).width, ==, (int) W / 2, "d");

    // Restore the full size for the frame loop below
    w = W;
    h = H;
    REQUIRE(pl_swapchain_resize(sw, &w, &h));
    REQUIRE_CMP(w, ==, (int) W, "d");

    // An HDR hint must switch the default 8-bit layer to a 10-bit format, and
    // an SDR hint must switch it back
    printf("- testing HDR colorspace hint\n");
    pl_swapchain_colorspace_hint(sw, &(struct pl_color_space) {
        .primaries = PL_COLOR_PRIM_BT_2020,
        .transfer = PL_COLOR_TRC_PQ,
    });
    struct pl_swapchain_frame hf = {0};
    REQUIRE(pl_swapchain_start_frame(sw, &hf));
    REQUIRE_CMP(hf.color_space.transfer, ==, PL_COLOR_TRC_PQ, "d");
    REQUIRE_CMP(hf.fbo->params.format->component_depth[0], ==, 10, "d");
    pl_tex_clear(gpu, hf.fbo, (float[4]) {0});
    REQUIRE(pl_swapchain_submit_frame(sw));
    pl_swapchain_swap_buffers(sw);
    pl_swapchain_colorspace_hint(sw, &pl_color_space_srgb);
    REQUIRE(pl_swapchain_start_frame(sw, &hf));
    REQUIRE_CMP(hf.fbo->params.format->component_depth[0], ==, 8, "d");
    REQUIRE(pl_swapchain_submit_frame(sw));
    pl_swapchain_swap_buffers(sw);

    // scRGB needs a float layer in extended linear sRGB, whose 1.0 is SDR
    // white, so the frame must be plain linear light rather than scRGB
    pl_swapchain_colorspace_hint(sw, &(struct pl_color_space) {
        .primaries = PL_COLOR_PRIM_BT_709,
        .transfer = PL_COLOR_TRC_SCRGB,
    });
    REQUIRE(pl_swapchain_start_frame(sw, &hf));
    REQUIRE_CMP(hf.color_space.transfer, ==, PL_COLOR_TRC_LINEAR, "d");
    REQUIRE_CMP(hf.fbo->params.format->type, ==, PL_FMT_FLOAT, "d");
    pl_tex_clear(gpu, hf.fbo, (float[4]) {0});
    // A caller pool opened mid-frame must survive submit_frame
    void *caller_pool = objc_autoreleasePoolPush();
    REQUIRE(pl_swapchain_submit_frame(sw));
    objc_autoreleasePoolPop(caller_pool);
    pl_swapchain_swap_buffers(sw);
    REQUIRE(!pl_gpu_is_failed(gpu));

    // Linear Display P3 has no layer colorspace of its own; whatever the layer
    // ends up tagged with is what the frame must report
    printf("- testing reported colorspace\n");
    pl_swapchain_colorspace_hint(sw, &(struct pl_color_space) {
        .primaries = PL_COLOR_PRIM_DISPLAY_P3,
        .transfer = PL_COLOR_TRC_LINEAR,
    });
    struct pl_swapchain_frame cf = {0};
    REQUIRE(pl_swapchain_start_frame(sw, &cf));
    REQUIRE_CMP(cf.color_space.primaries, ==, PL_COLOR_PRIM_DISPLAY_P3, "d");
    REQUIRE_CMP(cf.color_space.transfer, ==, PL_COLOR_TRC_SRGB, "d");

    // The framebuffer is blit_src, so a scaled blit out of it (which needs a
    // shader, hence sampling) must work too
    pl_fmt cfmt = cf.fbo->params.format;
    if (cfmt->type == PL_FMT_UNORM && cfmt->component_depth[0] == 8 &&
        (cfmt->caps & PL_FMT_CAP_HOST_READABLE))
    {
        printf("- testing scaled blit from the framebuffer\n");
        pl_tex_clear(gpu, cf.fbo, (float[4]) {1.0, 1.0, 1.0, 1.0});
        pl_tex half = pl_tex_create(gpu, pl_tex_params(
            .w = W / 2, .h = H / 2, .format = cfmt,
            .blit_dst = true, .host_readable = true,
        ));
        REQUIRE(half);
        pl_tex_clear(gpu, half, (float[4]) {0});
        pl_tex_blit(gpu, pl_tex_blit_params(.src = cf.fbo, .dst = half));
        size_t hsize = (size_t) (W / 2) * (H / 2) * cfmt->texel_size;
        uint8_t *hpx = malloc(hsize);
        REQUIRE(hpx);
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = half, .ptr = hpx)));
        for (size_t i = 0; i < hsize; i++)
            REQUIRE_CMP(hpx[i], ==, 0xFF, "u");
        free(hpx);
        pl_tex_destroy(gpu, &half);
    }

    pl_tex_clear(gpu, cf.fbo, (float[4]) {0});
    REQUIRE(pl_swapchain_submit_frame(sw));
    pl_swapchain_swap_buffers(sw);
    pl_swapchain_colorspace_hint(sw, &pl_color_space_srgb);

    // start_frame -> render -> submit -> swap, twice, to catch a lifecycle that
    // only works on the first frame. Each frame is cleared to a different
    // value and read back; comparing the two readbacks proves the drawable
    // really carries what was rendered, without assuming a pixel format.
    uint8_t *px[2] = {0};
    size_t px_size = 0;
    bool readback_ok[2] = {0};

    for (int frame = 0; frame < 2; frame++) {
        printf("- testing frame %d lifecycle\n", frame);
        struct pl_swapchain_frame f = {0};
        REQUIRE(pl_swapchain_start_frame(sw, &f));
        REQUIRE(f.fbo);
        REQUIRE_CMP(f.fbo->params.w, ==, (int) W, "d");
        REQUIRE_CMP(f.fbo->params.h, ==, (int) H, "d");
        REQUIRE(f.fbo->params.renderable);
        REQUIRE(f.fbo->params.blit_dst);

        float v = frame ? 0.25f : 0.75f;
        pl_tex_clear_ex(gpu, f.fbo, (union pl_clear_color){ .f = {v, v, v, 1.0f} });

        px_size = (size_t) W * H * f.fbo->params.format->texel_size;
        px[frame] = malloc(px_size);
        REQUIRE(px[frame]);
        memset(px[frame], 0, px_size);
        readback_ok[frame] = read_back_fbo(gpu, f.fbo, px[frame]);
        if (readback_ok[frame]) {
            printf("  cleared to %.2f, readback[0..3] = %u %u %u %u\n", v,
                   px[frame][0], px[frame][1], px[frame][2], px[frame][3]);
        } else {
            printf("  framebuffer not readable, skipping pixel check\n");
        }

        REQUIRE(pl_swapchain_submit_frame(sw));
        pl_swapchain_swap_buffers(sw);
        REQUIRE(!pl_gpu_is_failed(gpu));
    }

    if (readback_ok[0] && readback_ok[1]) {
        printf("- verifying the two frames differ\n");
        REQUIRE_CMP(memcmp(px[0], px[1], px_size), !=, 0, "d");
    }
    free(px[0]);
    free(px[1]);

    pl_swapchain_destroy(&sw);
    ((void (*)(id, SEL)) objc_msgSend)(layer, sel_registerName("release"));
    REQUIRE(!pl_gpu_is_failed(gpu));
}

// MTLPixelFormat values, from <Metal/MTLPixelFormat.h> (ObjC-only header)
enum {
    MTL_PIXEL_FORMAT_BGRA8_UNORM  = 80,
    MTL_PIXEL_FORMAT_RGB10A2_UNORM = 90,
    MTL_PIXEL_FORMAT_RGBA16_FLOAT  = 115,
};

// What a colorspace hint must leave on the CAMetalLayer, since that (and not
// the pl_swapchain_frame) is what the compositor actually uses
static void metal_swapchain_layer_tests(pl_metal metal)
{
    id layer = make_metal_layer(64, 48);
    if (!layer) {
        printf("metal_swapchain_layer_tests: CAMetalLayer unavailable"
               "... skipping this test\n");
        return;
    }

    pl_swapchain sw = pl_metal_create_swapchain(metal,
        pl_metal_swapchain_params(.layer = (void *) layer));
    REQUIRE(sw);
    printf("metal_swapchain_layer_tests:\n");

    SEL edr_sel = sel_registerName("wantsExtendedDynamicRangeContent");
    bool has_edr = class_respondsToSelector(object_getClass(layer), edr_sel);

    static const struct {
        enum pl_color_primaries prim;
        enum pl_color_transfer trc;
        unsigned long pixel_format;
        const CFStringRef *name;
        bool edr;
    } cases[] = {
        { PL_COLOR_PRIM_BT_2020,    PL_COLOR_TRC_PQ,     MTL_PIXEL_FORMAT_RGB10A2_UNORM,
          &kCGColorSpaceITUR_2100_PQ,         true },
        { PL_COLOR_PRIM_BT_2020,    PL_COLOR_TRC_HLG,    MTL_PIXEL_FORMAT_RGB10A2_UNORM,
          &kCGColorSpaceITUR_2100_HLG,        true },
        { PL_COLOR_PRIM_DISPLAY_P3, PL_COLOR_TRC_PQ,     MTL_PIXEL_FORMAT_RGB10A2_UNORM,
          &kCGColorSpaceDisplayP3_PQ,         true },
        { PL_COLOR_PRIM_DISPLAY_P3, PL_COLOR_TRC_HLG,    MTL_PIXEL_FORMAT_RGB10A2_UNORM,
          &kCGColorSpaceDisplayP3_HLG,        true },
        { PL_COLOR_PRIM_BT_709,     PL_COLOR_TRC_SCRGB,  MTL_PIXEL_FORMAT_RGBA16_FLOAT,
          &kCGColorSpaceExtendedLinearSRGB,   true },
        { PL_COLOR_PRIM_BT_709,     PL_COLOR_TRC_LINEAR, MTL_PIXEL_FORMAT_RGBA16_FLOAT,
          &kCGColorSpaceLinearSRGB,           false },
        { PL_COLOR_PRIM_DISPLAY_P3, PL_COLOR_TRC_SRGB,   MTL_PIXEL_FORMAT_BGRA8_UNORM,
          &kCGColorSpaceDisplayP3,            false },
        { PL_COLOR_PRIM_BT_709,     PL_COLOR_TRC_SRGB,   MTL_PIXEL_FORMAT_BGRA8_UNORM,
          &kCGColorSpaceSRGB,                 false },
    };

    for (int i = 0; i < PL_ARRAY_SIZE(cases); i++) {
        pl_swapchain_colorspace_hint(sw, &(struct pl_color_space) {
            .primaries = cases[i].prim,
            .transfer = cases[i].trc,
        });

        unsigned long pf = ((unsigned long (*)(id, SEL)) objc_msgSend)(
            layer, sel_registerName("pixelFormat"));
        CGColorSpaceRef cs = ((CGColorSpaceRef (*)(id, SEL)) objc_msgSend)(
            layer, sel_registerName("colorspace"));
        REQUIRE(cs);
        CFStringRef name = CGColorSpaceCopyName(cs);
        REQUIRE(name);
        char buf[128] = "?";
        CFStringGetCString(name, buf, sizeof(buf), kCFStringEncodingUTF8);
        printf("  case %d -> pixelFormat %lu, %s\n", i, pf, buf);

        REQUIRE_CMP(pf, ==, cases[i].pixel_format, "lu");
        REQUIRE(CFEqual(name, *cases[i].name));
        CFRelease(name);
        if (has_edr) {
            bool edr = ((BOOL (*)(id, SEL)) objc_msgSend)(layer, edr_sel);
            REQUIRE_CMP(edr, ==, cases[i].edr, "d");
        }
    }

    pl_swapchain_destroy(&sw);
    ((void (*)(id, SEL)) objc_msgSend)(layer, sel_registerName("release"));
}

// Reference HDR encodings, in double precision and straight from the
// standards, so that they share nothing with the GLSL under test

// SMPTE ST 2084, cd/m^2 <-> signal
static const double PQ_M1 = 2610.0 / 16384, PQ_M2 = 2523.0 / 4096 * 128,
                    PQ_C1 = 3424.0 / 4096, PQ_C2 = 2413.0 / 4096 * 32,
                    PQ_C3 = 2392.0 / 4096 * 32;

static double ref_pq_encode(double nits)
{
    double y = pow(fmax(nits, 0) / 10000, PQ_M1);
    return pow((PQ_C1 + PQ_C2 * y) / (1 + PQ_C3 * y), PQ_M2);
}

static double ref_pq_decode(double v)
{
    double p = pow(v, 1 / PQ_M2);
    return 10000 * pow(fmax(p - PQ_C1, 0) / (PQ_C2 - PQ_C3 * p), 1 / PQ_M1);
}

// ITU-R BT.2100 HLG, for a display with peak `lw` and black `lb` (cd/m^2)
static const double HLG_REF_A = 0.17883277, HLG_REF_B = 1 - 4 * 0.17883277;
#define HLG_REF_C (0.5 - HLG_REF_A * log(4 * HLG_REF_A))

static double ref_hlg_gamma(double lw)
{
    return 1.2 * pow(1.111, log2(lw / 1000)); // table 5, note 5f
}

static double ref_hlg_beta(double lw, double lb)
{
    return sqrt(3 * pow(lb / lw, 1 / ref_hlg_gamma(lw)));
}

static double ref_bt2020_luma(const double rgb[3])
{
    return 0.2627 * rgb[0] + 0.6780 * rgb[1] + 0.0593 * rgb[2];
}

// Display light -> signal: inverse OOTF, OETF, then undo the black lift
static void ref_hlg_encode(const double nits[3], double lw, double lb, double out[3])
{
    double g = ref_hlg_gamma(lw), beta = ref_hlg_beta(lw, lb);
    double yd = ref_bt2020_luma(nits);
    for (int c = 0; c < 3; c++) {
        double e = yd > 0 ? nits[c] / lw * pow(yd / lw, (1 - g) / g) : 0;
        double v = e <= 1.0 / 12 ? sqrt(3 * e)
                                 : HLG_REF_A * log(12 * e - HLG_REF_B) + HLG_REF_C;
        out[c] = (v - beta) / (1 - beta);
    }
}

// Signal -> display light, the BT.2100 HLG EOTF
static void ref_hlg_decode(const double sig[3], double lw, double lb, double out[3])
{
    double g = ref_hlg_gamma(lw), beta = ref_hlg_beta(lw, lb), e[3];
    for (int c = 0; c < 3; c++) {
        double v = fmax(0, (1 - beta) * sig[c] + beta);
        e[c] = v <= 0.5 ? v * v / 3 : (exp((v - HLG_REF_C) / HLG_REF_A) + HLG_REF_B) / 12;
    }
    double ys = ref_bt2020_luma(e);
    for (int c = 0; c < 3; c++)
        out[c] = ys > 0 ? lw * pow(ys, g - 1) * e[c] : 0;
}

// ITU-R BT.2124 ΔE ITP between two BT.2020 colours in display light (cd/m^2).
// 1.0 is about one just noticeable difference.
static double ref_delta_e_itp(const double a[3], const double b[3])
{
    double itp[2][3];
    const double *rgb[2] = { a, b };
    for (int i = 0; i < 2; i++) {
        const double *x = rgb[i];
        double lms[3] = { // BT.2100 table 7
            ref_pq_encode((1688 * x[0] + 2146 * x[1] +  262 * x[2]) / 4096),
            ref_pq_encode(( 683 * x[0] + 2951 * x[1] +  462 * x[2]) / 4096),
            ref_pq_encode((  99 * x[0] +  309 * x[1] + 3688 * x[2]) / 4096),
        };
        itp[i][0] = 0.5 * lms[0] + 0.5 * lms[1];
        itp[i][1] = 0.5 * (6610 * lms[0] - 13613 * lms[1] + 7003 * lms[2]) / 4096;
        itp[i][2] = (17933 * lms[0] - 17390 * lms[1] - 543 * lms[2]) / 4096;
    }
    double di = itp[0][0] - itp[1][0], dt = itp[0][1] - itp[1][1],
           dp = itp[0][2] - itp[1][2];
    return 720 * sqrt(di * di + dt * dt + dp * dp);
}

// BT.709 -> BT.2020 for linear light, ITU-R BT.2087 equation 4
static void ref_bt709_to_bt2020(const double in[3], double out[3])
{
    static const double m[3][3] = {
        { 0.6274, 0.3293, 0.0433 },
        { 0.0691, 0.9195, 0.0114 },
        { 0.0164, 0.0880, 0.8956 },
    };
    for (int r = 0; r < 3; r++)
        out[r] = m[r][0] * in[0] + m[r][1] * in[1] + m[r][2] * in[2];
}

// Test patches in linear light (cd/m^2), each set in its own primaries
static const double hdr_patches_2020[][3] = {
    {    0,    0,    0 }, {  0.1,  0.1,  0.1 }, {    1,    1,    1 },
    {   10,   10,   10 }, {  100,  100,  100 }, {  203,  203,  203 },
    {  500,  500,  500 }, { 1000, 1000, 1000 },
    // Saturated colours stay at 500 cd/m^2: brighter than that, HLG's
    // luminance-based OOTF pushes pure blue out of the signal range
    {  500,    0,    0 }, {    0,  500,    0 }, {    0,    0,  500 },
    {    0,  500,  500 }, {  500,    0,  500 }, {  500,  500,    0 },
    {  100,   20,    5 }, {   20,  100,   50 },
};

static const double hdr_patches_709[][3] = {
    { 203,   0,   0 }, {   0, 203,   0 }, {   0,   0, 203 },
    { 203, 203,   0 }, {   0, 203, 203 }, { 203,   0, 203 },
    { 203, 203, 203 }, {  50,  30,  10 },
};

#define HDR_KAT_MAX_PATCHES 16
#define HDR_KAT_LW 1000.0

// Unpack one pixel of an RGB10A2Unorm texture: R in the low bits
static void unpack_rgb10a2(uint32_t px, int out[3])
{
    for (int c = 0; c < 3; c++)
        out[c] = (px >> (10 * c)) & 0x3FF;
}

struct hdr_kat_result {
    int max_code_diff;
    double max_delta_e;
};

// Render one patch set (a single row) into the top left of `fbo` with the
// colorimetry of `dst`
static void hdr_kat_render(pl_gpu gpu, pl_renderer rr, pl_tex fbo,
                           const struct pl_color_space *dst,
                           enum pl_color_primaries src_prim,
                           const double (*patches)[3], int num)
{
    REQUIRE_CMP(num, <=, HDR_KAT_MAX_PATCHES, "d");
    pl_fmt src_fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 32, 32, PL_FMT_CAP_SAMPLEABLE);
    REQUIRE(src_fmt);

    float src_px[HDR_KAT_MAX_PATCHES][4];
    for (int i = 0; i < num; i++) {
        for (int c = 0; c < 3; c++)
            src_px[i][c] = patches[i][c] / PL_COLOR_SDR_WHITE; // NORM scale
        src_px[i][3] = 1.0f;
    }

    pl_tex src = pl_tex_create(gpu, pl_tex_params(
        .w = num, .h = 1, .format = src_fmt,
        .sampleable = true, .initial_data = src_px,
    ));
    REQUIRE(src);

    struct pl_frame image = {
        .num_planes = 1,
        .planes = {{ .texture = src, .components = 3, .component_mapping = {0, 1, 2} }},
        .repr = pl_color_repr_rgb,
        .color = {
            .primaries = src_prim,
            .transfer = PL_COLOR_TRC_LINEAR,
            .hdr = { .min_luma = PL_COLOR_HDR_BLACK, .max_luma = HDR_KAT_LW },
        },
    };

    struct pl_frame target = {
        .num_planes = 1,
        .planes = {{ .texture = fbo, .components = 3, .component_mapping = {0, 1, 2} }},
        .repr = pl_color_repr_rgb,
        .color = *dst,
        .crop = { 0, 0, num, 1 },
    };

    // Nothing in range needs tone or gamut mapping, so anything but the plain
    // transfer and primaries conversion would show up as an error
    struct pl_render_params params = pl_render_default_params;
    params.dither_params = NULL;
    params.peak_detect_params = NULL;
    REQUIRE(pl_render_image(rr, &image, &target, &params));
    pl_tex_destroy(gpu, &src);
}

// Render a patch set as above, then compare every pixel against the reference
static void hdr_kat_run(pl_gpu gpu, pl_renderer rr, pl_tex fbo,
                        const struct pl_color_space *dst,
                        enum pl_color_primaries src_prim,
                        const double (*patches)[3], int num,
                        struct hdr_kat_result *res)
{
    hdr_kat_render(gpu, rr, fbo, dst, src_prim, patches, num);
    uint32_t *out = malloc((size_t) fbo->params.w * fbo->params.h * sizeof(uint32_t));
    REQUIRE(out);
    REQUIRE(read_back_fbo(gpu, fbo, (uint8_t *) out));

    bool pq = dst->transfer == PL_COLOR_TRC_PQ;
    for (int i = 0; i < num; i++) {
        double want[3];
        if (src_prim == PL_COLOR_PRIM_BT_709) {
            ref_bt709_to_bt2020(patches[i], want);
        } else {
            memcpy(want, patches[i], sizeof(want));
        }

        double sig[3], got[3], got_sig[3];
        if (pq) {
            for (int c = 0; c < 3; c++)
                sig[c] = ref_pq_encode(want[c]);
        } else {
            ref_hlg_encode(want, dst->hdr.max_luma, dst->hdr.min_luma, sig);
        }

        // An out of range reference would only test the clipping
        for (int c = 0; c < 3; c++)
            REQUIRE(sig[c] > -1e-3 && sig[c] < 1 + 1e-9);

        int code[3];
        unpack_rgb10a2(out[i], code);
        for (int c = 0; c < 3; c++) {
            int ref = (int) lrint(PL_CLAMP(sig[c], 0.0, 1.0) * 1023);
            int diff = abs(code[c] - ref);
            if (diff > res->max_code_diff)
                res->max_code_diff = diff;
            if (diff > 1) {
                printf("  patch %d channel %d: got code %d, want %d (%.3f cd/m^2)\n",
                       i, c, code[c], ref, want[c]);
            }
            got_sig[c] = code[c] / 1023.0;
        }

        if (pq) {
            for (int c = 0; c < 3; c++)
                got[c] = ref_pq_decode(got_sig[c]);
        } else {
            ref_hlg_decode(got_sig, dst->hdr.max_luma, dst->hdr.min_luma, got);
        }
        double de = ref_delta_e_itp(got, want);
        if (de > res->max_delta_e)
            res->max_delta_e = de;
    }

    free(out);
}

// Gates: within one 10-bit code value, and below one JND everywhere
static void hdr_kat_check(const char *what, const struct hdr_kat_result *res)
{
    printf("  %-24s max code diff %d, max ΔE ITP %.3f\n", what,
           res->max_code_diff, res->max_delta_e);
    REQUIRE_CMP(res->max_code_diff, <=, 1, "d");
    REQUIRE_CMP(res->max_delta_e, <, 1.0, "f");
}

static const struct pl_color_space hdr_kat_spaces[] = {
    {
        .primaries = PL_COLOR_PRIM_BT_2020,
        .transfer = PL_COLOR_TRC_PQ,
        .hdr = { .min_luma = PL_COLOR_HDR_BLACK, .max_luma = 10000 },
    }, {
        .primaries = PL_COLOR_PRIM_BT_2020,
        .transfer = PL_COLOR_TRC_HLG,
        .hdr = { .min_luma = PL_COLOR_HDR_BLACK, .max_luma = HDR_KAT_LW },
    },
};

// Known-answer test of the PQ and HLG output encoding, against an offscreen
// 10-bit target
static void metal_hdr_output_tests(pl_gpu gpu)
{
    pl_fmt fmt = pl_find_named_fmt(gpu, "rgb10a2");
    if (!fmt || !(fmt->caps & PL_FMT_CAP_RENDERABLE) ||
        !(fmt->caps & PL_FMT_CAP_HOST_READABLE))
    {
        printf("metal_hdr_output_tests: no renderable, readable rgb10a2"
               "... skipping this test\n");
        return;
    }

    printf("metal_hdr_output_tests:\n");
    pl_tex fbo = pl_tex_create(gpu, pl_tex_params(
        .w = HDR_KAT_MAX_PATCHES, .h = 1, .format = fmt,
        .renderable = true, .blit_src = true, .blit_dst = true,
        .host_readable = true,
    ));
    REQUIRE(fbo);

    // Pin down the packing that unpack_rgb10a2() assumes
    uint32_t px[HDR_KAT_MAX_PATCHES];
    pl_tex_clear(gpu, fbo, (float[4]) { 1.0, 0.0, 0.0, 1.0 });
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = fbo, .ptr = px)));
    REQUIRE_CMP(px[0], ==, 0x3FFu | (3u << 30), "x");

    pl_renderer rr = pl_renderer_create(gpu->log, gpu);
    REQUIRE(rr);
    for (int i = 0; i < PL_ARRAY_SIZE(hdr_kat_spaces); i++) {
        const struct pl_color_space *dst = &hdr_kat_spaces[i];
        bool pq = dst->transfer == PL_COLOR_TRC_PQ;

        struct hdr_kat_result res = {0};
        hdr_kat_run(gpu, rr, fbo, dst, PL_COLOR_PRIM_BT_2020, hdr_patches_2020,
                    PL_ARRAY_SIZE(hdr_patches_2020), &res);
        hdr_kat_check(pq ? "BT.2020 -> PQ" : "BT.2020 -> HLG", &res);

        res = (struct hdr_kat_result) {0};
        hdr_kat_run(gpu, rr, fbo, dst, PL_COLOR_PRIM_BT_709, hdr_patches_709,
                    PL_ARRAY_SIZE(hdr_patches_709), &res);
        hdr_kat_check(pq ? "BT.709 -> PQ" : "BT.709 -> HLG", &res);
    }

    pl_renderer_destroy(&rr);
    pl_tex_destroy(gpu, &fbo);
}

// The same known-answer test, but rendered into real drawables, which covers
// everything up to the point where the frame is handed to the compositor
static void metal_swapchain_hdr_tests(pl_metal metal)
{
    pl_gpu gpu = metal->gpu;
    id layer = make_metal_layer(64, 48);
    if (!layer) {
        printf("metal_swapchain_hdr_tests: CAMetalLayer unavailable"
               "... skipping this test\n");
        return;
    }

    pl_swapchain sw = pl_metal_create_swapchain(metal,
        pl_metal_swapchain_params(.layer = (void *) layer));
    REQUIRE(sw);
    pl_renderer rr = pl_renderer_create(gpu->log, gpu);
    REQUIRE(rr);
    printf("metal_swapchain_hdr_tests:\n");

    for (int i = 0; i < PL_ARRAY_SIZE(hdr_kat_spaces); i++) {
        pl_swapchain_colorspace_hint(sw, &hdr_kat_spaces[i]);
        struct pl_swapchain_frame f = {0};
        REQUIRE(pl_swapchain_start_frame(sw, &f));
        REQUIRE_CMP(f.color_space.transfer, ==, hdr_kat_spaces[i].transfer, "d");
        REQUIRE(f.fbo->params.blit_src);
        REQUIRE(f.fbo->params.format == pl_find_named_fmt(gpu, "rgb10a2"));

        bool pq = f.color_space.transfer == PL_COLOR_TRC_PQ;
        struct hdr_kat_result res = {0};
        hdr_kat_run(gpu, rr, f.fbo, &f.color_space, PL_COLOR_PRIM_BT_2020,
                    hdr_patches_2020, PL_ARRAY_SIZE(hdr_patches_2020), &res);
        hdr_kat_check(pq ? "drawable, PQ" : "drawable, HLG", &res);

        REQUIRE(pl_swapchain_submit_frame(sw));
        pl_swapchain_swap_buffers(sw);
        REQUIRE(!pl_gpu_is_failed(gpu));
    }

    // scRGB lands in an extended linear sRGB layer. What a value there means
    // is up to Apple, so take the reference from CoreGraphics itself: the
    // ST 2084 encoding of each patch, matched into the layer's colorspace.
    static const double scrgb_patches[][3] = {
        {    0,    0,    0 }, {    1,    1,    1 }, {  100,  100,  100 },
        {  203,  203,  203 }, { 1000, 1000, 1000 }, {  203,    0,    0 },
        {    0,  203,    0 }, {    0,    0,  203 }, {   50,   30,   10 },
    };
    const int num = PL_ARRAY_SIZE(scrgb_patches);

    pl_swapchain_colorspace_hint(sw, &(struct pl_color_space) {
        .primaries = PL_COLOR_PRIM_BT_709,
        .transfer = PL_COLOR_TRC_SCRGB,
    });
    struct pl_swapchain_frame f = {0};
    REQUIRE(pl_swapchain_start_frame(sw, &f));
    pl_fmt ffmt = f.fbo->params.format;
    REQUIRE(ffmt->type == PL_FMT_FLOAT && ffmt->component_depth[0] == 16);
    REQUIRE(ffmt->texel_size == 8 && (ffmt->caps & PL_FMT_CAP_HOST_READABLE));
    hdr_kat_render(gpu, rr, f.fbo, &f.color_space, PL_COLOR_PRIM_BT_709,
                   scrgb_patches, num);

    __fp16 *out = malloc((size_t) f.fbo->params.w * f.fbo->params.h * 8);
    REQUIRE(out);
    REQUIRE(read_back_fbo(gpu, f.fbo, (uint8_t *) out));

    CGColorSpaceRef pq_cs = CGColorSpaceCreateWithName(kCGColorSpaceITUR_2100_PQ);
    CGColorSpaceRef ext_cs = CGColorSpaceCreateWithName(kCGColorSpaceExtendedLinearSRGB);
    REQUIRE(pq_cs && ext_cs);
    double max_err = 0;
    for (int i = 0; i < num; i++) {
        double rgb2020[3];
        ref_bt709_to_bt2020(scrgb_patches[i], rgb2020);
        CGFloat comps[4] = { 0, 0, 0, 1 };
        for (int c = 0; c < 3; c++)
            comps[c] = ref_pq_encode(rgb2020[c]);
        CGColorRef pq_col = CGColorCreate(pq_cs, comps);
        CGColorRef ext_col = CGColorCreateCopyByMatchingToColorSpace(
            ext_cs, kCGRenderingIntentDefault, pq_col, NULL);
        REQUIRE(ext_col);
        const CGFloat *want = CGColorGetComponents(ext_col);
        for (int c = 0; c < 3; c++) {
            double got = out[i * 4 + c];
            // fp16 keeps 11 significant bits, and BT.2087 rounds to 4 digits,
            // which leaves about 1e-4 of crosstalk around zero
            double tol = 2e-3 * fabs(want[c]) + 1e-4;
            double err = fabs(got - want[c]);
            max_err = fmax(max_err, err / tol);
            if (err > tol) {
                printf("  patch %d channel %d: got %f, CoreGraphics says %f\n",
                       i, c, got, (double) want[c]);
            }
        }
        CGColorRelease(ext_col);
        CGColorRelease(pq_col);
    }
    printf("  %-24s max error %.2f of tolerance\n", "drawable, scRGB", max_err);
    REQUIRE_CMP(max_err, <=, 1.0, "f");
    CGColorSpaceRelease(ext_cs);
    CGColorSpaceRelease(pq_cs);
    free(out);

    REQUIRE(pl_swapchain_submit_frame(sw));
    pl_swapchain_swap_buffers(sw);
    REQUIRE(!pl_gpu_is_failed(gpu));

    pl_renderer_destroy(&rr);
    pl_swapchain_destroy(&sw);
    ((void (*)(id, SEL)) objc_msgSend)(layer, sel_registerName("release"));
}

// Each of these reproduces a hazard that the shared suite cannot see, because
// it synchronizes (reads back) between most operations.

static pl_tex hazard_tex(pl_gpu gpu, pl_fmt fmt, int w, int h, bool host)
{
    pl_tex tex = pl_tex_create(gpu, pl_tex_params(
        .w = w, .h = h, .format = fmt,
        .sampleable = true, .blit_src = true, .blit_dst = true,
        .host_writable = host, .host_readable = host,
    ));
    REQUIRE(tex);
    return tex;
}

static void metal_hazard_tests(pl_gpu gpu)
{
    printf("metal_hazard_tests:\n");

    // pl_buf_write() must not overtake a queued copy that reads the old data
    printf("- buf write after pending copy\n");
    pl_buf a = pl_buf_create(gpu, pl_buf_params(
        .size = 16, .host_writable = true, .host_readable = true));
    pl_buf b = pl_buf_create(gpu, pl_buf_params(
        .size = 16, .host_writable = true, .host_readable = true));
    REQUIRE(a && b);
    static const uint32_t old_data[4] = {1, 2, 3, 4}, new_data[4] = {5, 6, 7, 8};
    uint32_t out[4] = {0};
    pl_buf_write(gpu, a, 0, old_data, sizeof(old_data));
    pl_buf_copy(gpu, b, 0, a, 0, sizeof(old_data));
    pl_buf_write(gpu, a, 0, new_data, sizeof(new_data));
    REQUIRE(pl_buf_read(gpu, b, 0, out, sizeof(out)));
    REQUIRE_MEMEQ(out, old_data, sizeof(out));
    pl_buf_destroy(gpu, &a);
    pl_buf_destroy(gpu, &b);

    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_UNORM, 4, 8, 8,
                             PL_FMT_CAP_BLITTABLE | PL_FMT_CAP_HOST_READABLE);
    REQUIRE(fmt);

    // pl_tex_upload() must not overtake a queued GPU read of the old data.
    // `mid` has no host access, so both blits take the GPU path.
    printf("- tex upload after pending blit\n");
    enum { W = 4, H = 4 };
    uint8_t old_px[W * H * 4], new_px[W * H * 4], px[W * H * 4];
    for (int i = 0; i < W * H * 4; i++) {
        old_px[i] = (uint8_t) i;
        new_px[i] = (uint8_t) (255 - i);
    }
    pl_tex src = hazard_tex(gpu, fmt, W, H, true);
    pl_tex mid = hazard_tex(gpu, fmt, W, H, false);
    pl_tex dst = hazard_tex(gpu, fmt, W, H, true);
    REQUIRE(pl_tex_upload(gpu, pl_tex_transfer_params(.tex = src, .ptr = old_px)));
    pl_tex_blit(gpu, pl_tex_blit_params(.src = src, .dst = mid));
    REQUIRE(pl_tex_upload(gpu, pl_tex_transfer_params(.tex = src, .ptr = new_px)));
    pl_tex_blit(gpu, pl_tex_blit_params(.src = mid, .dst = dst));
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = dst, .ptr = px)));
    REQUIRE_MEMEQ(px, old_px, sizeof(px));

    // Streamed vertex data must survive an asynchronous flush: draw the left
    // half into one target, flush without waiting, then the right half into
    // another. The first target must still only have its left half filled.
    printf("- streamed vertices across a flush\n");
    pl_fmt ffmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 16, 32,
                              PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_BLITTABLE |
                              PL_FMT_CAP_HOST_READABLE);
    REQUIRE(ffmt);
    enum { FW = 16, FH = 4 };
    pl_tex fbo[2];
    for (int i = 0; i < 2; i++) {
        fbo[i] = pl_tex_create(gpu, pl_tex_params(
            .w = FW, .h = FH, .format = ffmt,
            .renderable = true, .blit_dst = true, .host_readable = true));
        REQUIRE(fbo[i]);
        pl_tex_clear(gpu, fbo[i], (float[4]) {0});
    }
    pl_dispatch dp = pl_dispatch_create(gpu->log, gpu);
    REQUIRE(dp);
    for (int i = 0; i < 2; i++) {
        pl_shader sh = pl_dispatch_begin(dp);
        sh_describe(sh, "solid");
        GLSL("vec4 color = vec4(1.0); \n");
        sh->output = PL_SHADER_SIG_COLOR;
        REQUIRE(pl_dispatch_finish(dp, pl_dispatch_params(
            .shader = &sh,
            .target = fbo[i],
            .rect = { i ? FW / 2 : 0, 0, i ? FW : FW / 2, FH },
        )));
        pl_gpu_flush(gpu);
    }
    float fpx[FW * FH * 4];
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = fbo[0], .ptr = fpx)));
    for (int x = 0; x < FW; x++)
        REQUIRE_FEQ(fpx[x * 4], x < FW / 2 ? 1.0 : 0.0, 1e-6);
    pl_dispatch_destroy(&dp);
    for (int i = 0; i < 2; i++)
        pl_tex_destroy(gpu, &fbo[i]);

    // Flipped and scaled blits, on both the CPU (both textures host-visible)
    // and GPU (one side without host access) paths
    printf("- flipped and scaled blits\n");
    for (int gpu_path = 0; gpu_path <= 1; gpu_path++) {
        pl_tex flip = hazard_tex(gpu, fmt, W, H, !gpu_path);
        pl_tex big = hazard_tex(gpu, fmt, 2 * W, 2 * H, !gpu_path);
        pl_tex big_out = hazard_tex(gpu, fmt, 2 * W, 2 * H, true);
        REQUIRE(pl_tex_upload(gpu, pl_tex_transfer_params(.tex = src, .ptr = old_px)));

        // Flip the inner 2x2 region; everything outside dst_rc must survive
        static const uint8_t zero[4] = {0};
        pl_tex_clear(gpu, flip, (float[4]) {0});
        pl_tex_blit(gpu, pl_tex_blit_params(
            .src = src, .dst = flip,
            .src_rc = { .x0 = 1, .y0 = 1, .x1 = 3, .y1 = 3 },
            .dst_rc = { .x0 = 3, .y0 = 3, .x1 = 1, .y1 = 1 },
        ));
        pl_tex_blit(gpu, pl_tex_blit_params(.src = flip, .dst = dst));
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = dst, .ptr = px)));
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                bool inside = x >= 1 && x < 3 && y >= 1 && y < 3;
                REQUIRE_MEMEQ(&px[(y * W + x) * 4], inside
                    ? &old_px[((3 - y) * W + (3 - x)) * 4] : zero, 4);
            }
        }

        uint8_t bpx[4 * W * H * 4];
        pl_tex_blit(gpu, pl_tex_blit_params(
            .src = src, .dst = big, .sample_mode = PL_TEX_SAMPLE_NEAREST));
        pl_tex_blit(gpu, pl_tex_blit_params(.src = big, .dst = big_out));
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = big_out, .ptr = bpx)));
        for (int y = 0; y < 2 * H; y++)
            for (int x = 0; x < 2 * W; x++)
                REQUIRE_MEMEQ(&bpx[(y * 2 * W + x) * 4],
                              &old_px[((y / 2) * W + x / 2) * 4], 4);

        pl_tex_destroy(gpu, &flip);
        pl_tex_destroy(gpu, &big);
        pl_tex_destroy(gpu, &big_out);
    }

    // A blit-only destination (not renderable) must still be clearable
    printf("- clear a blit-only texture\n");
    pl_tex_clear(gpu, mid, (float[4]) {1.0, 0.0, 0.0, 1.0});
    pl_tex_blit(gpu, pl_tex_blit_params(.src = mid, .dst = dst));
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = dst, .ptr = px)));
    for (int i = 0; i < W * H; i++)
        REQUIRE_MEMEQ(&px[i * 4], ((uint8_t[4]) {255, 0, 0, 255}), 4);

    pl_tex_destroy(gpu, &src);
    pl_tex_destroy(gpu, &mid);
    pl_tex_destroy(gpu, &dst);
}

// A compute pass recreated from a populated pl_cache must hit that cache and
// still produce the same output.
static void metal_cache_tests(pl_gpu gpu)
{
    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 16, 32,
                             PL_FMT_CAP_STORABLE | PL_FMT_CAP_BLITTABLE |
                             PL_FMT_CAP_HOST_READABLE);
    if (!fmt || !gpu->glsl.compute) {
        printf("metal_cache_tests: no storable format... skipping this test\n");
        return;
    }

    printf("metal_cache_tests:\n");
    enum { W = 16, H = 16 };
    static float src_px[W * H * 4];
    for (int i = 0; i < W * H * 4; i++)
        src_px[i] = (float) i / (W * H * 4);

    pl_tex src = pl_tex_create(gpu, pl_tex_params(
        .w = W, .h = H, .format = fmt, .sampleable = true,
        .initial_data = src_px));
    pl_tex fbo = pl_tex_create(gpu, pl_tex_params(
        .w = W, .h = H, .format = fmt, .storable = true, .blit_dst = true,
        .host_readable = true));
    REQUIRE(src && fbo);

    pl_cache cache = pl_cache_create(pl_cache_params( .log = gpu->log ));
    pl_gpu_set_cache(gpu, cache);
    int objects[3];
    static float out[W * H * 4];
    for (int i = 0; i < 3; i++) {
        pl_dispatch dp = pl_dispatch_create(gpu->log, gpu);
        REQUIRE(dp);
        pl_tex_clear(gpu, fbo, (float[4]) {0});
        pl_shader sh = pl_dispatch_begin(dp);
        sh->type = SH_COMPUTE;
        sh->group_size[0] = 8;
        sh->group_size[1] = 8;
        REQUIRE(pl_shader_sample_direct(sh, pl_sample_src( .tex = src )));
        REQUIRE(pl_dispatch_finish(dp, pl_dispatch_params(
            .shader = &sh, .target = fbo)));
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = fbo, .ptr = out)));
        REQUIRE_CMP(img_cmp_count_above(out, src_px, W * H * 4, 1e-6f), ==, 0, "d");
        objects[i] = pl_cache_objects(cache);
        pl_dispatch_destroy(&dp);
    }
    // later pass creations must be served from the cache, and keep it intact
    REQUIRE_CMP(objects[0], >, 0, "d");
    REQUIRE_CMP(objects[1], ==, objects[0], "d");
    REQUIRE_CMP(objects[2], ==, objects[0], "d");

    pl_gpu_set_cache(gpu, NULL);
    pl_cache_destroy(&cache);
    pl_tex_destroy(gpu, &src);
    pl_tex_destroy(gpu, &fbo);
}

static void metal_transfer_tests(pl_gpu gpu)
{
    printf("metal_transfer_tests:\n");

    // A full-width, partial-height region of a 3D texture: rows match the
    // tight layout but slices do not, so the slice stride must be honoured
    printf("- partial 3D upload\n");
    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_UNORM, 4, 8, 8, PL_FMT_CAP_HOST_READABLE);
    REQUIRE(fmt);
    enum { W = 4, H = 4, D = 2 };
    uint8_t data[D][H][W][4], out[D][H][W][4];
    for (int i = 0; i < (int) sizeof(data); i++)
        ((uint8_t *) data)[i] = (uint8_t) (i + 1);
    pl_tex tex = pl_tex_create(gpu, pl_tex_params(
        .w = W, .h = H, .d = D, .format = fmt,
        .host_writable = true, .host_readable = true,
        .initial_data = (uint8_t[D * H * W * 4]) {0},
    ));
    REQUIRE(tex);
    REQUIRE(pl_tex_upload(gpu, pl_tex_transfer_params(
        .tex = tex, .ptr = data,
        .rc = { 0, 0, 0, W, H / 2, D },
        .row_pitch = sizeof(data[0][0]),
        .depth_pitch = sizeof(data[0]),
    )));
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = tex, .ptr = out)));
    for (int z = 0; z < D; z++) {
        for (int y = 0; y < H; y++) {
            if (y < H / 2)
                REQUIRE_MEMEQ(out[z][y], data[z][y], sizeof(data[z][y]));
            else
                REQUIRE_MEMEQ(out[z][y], ((uint8_t[W][4]) {0}), sizeof(out[z][y]));
        }
    }
    pl_tex_destroy(gpu, &tex);

    // The same through a buffer at an offset. Overwriting the buffer right
    // after must not change what was uploaded from it.
    printf("- partial 3D upload from a buffer\n");
    enum { OFFSET = 16 };
    pl_buf buf = pl_buf_create(gpu, pl_buf_params(
        .size = OFFSET + sizeof(data), .host_writable = true));
    REQUIRE(buf);
    pl_buf_write(gpu, buf, OFFSET, data, sizeof(data));
    tex = pl_tex_create(gpu, pl_tex_params(
        .w = W, .h = H, .d = D, .format = fmt,
        .host_writable = true, .host_readable = true,
        .initial_data = (uint8_t[D * H * W * 4]) {0},
    ));
    REQUIRE(tex);
    REQUIRE(pl_tex_upload(gpu, pl_tex_transfer_params(
        .tex = tex, .buf = buf, .buf_offset = OFFSET,
        .rc = { 0, 0, 0, W, H / 2, D },
        .row_pitch = sizeof(data[0][0]),
        .depth_pitch = sizeof(data[0]),
    )));
    pl_buf_write(gpu, buf, OFFSET, (uint8_t[sizeof(data)]) {0}, sizeof(data));
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = tex, .ptr = out)));
    for (int z = 0; z < D; z++) {
        for (int y = 0; y < H; y++) {
            if (y < H / 2)
                REQUIRE_MEMEQ(out[z][y], data[z][y], sizeof(data[z][y]));
            else
                REQUIRE_MEMEQ(out[z][y], ((uint8_t[W][4]) {0}), sizeof(out[z][y]));
        }
    }
    pl_tex_destroy(gpu, &tex);
    pl_buf_destroy(gpu, &buf);

    // A non-zero clear of a packed format must not depend on host access
    printf("- packed format clear\n");
    pl_fmt packed = pl_find_named_fmt(gpu, "rgb10a2");
    REQUIRE(packed);
    pl_tex ptex = pl_tex_create(gpu, pl_tex_params(
        .w = W, .h = H, .format = packed,
        .blit_dst = true, .host_readable = true,
    ));
    REQUIRE(ptex);
    pl_tex_clear(gpu, ptex, (float[4]) {1.0, 0.0, 0.0, 1.0});
    uint32_t pout[W * H];
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = ptex, .ptr = pout)));
    for (int i = 0; i < W * H; i++)
        REQUIRE_CMP(pout[i], ==, 0xC00003FFu, "x"); // R = 1023, A = 3
    pl_tex_destroy(gpu, &ptex);
}

// Wrapped textures must only claim what the underlying MTLTexture allows
static void metal_wrap_tests(pl_metal metal)
{
    pl_gpu gpu = metal->gpu;
    printf("metal_wrap_tests:\n");
    printf("- wrapping a private, sample-only texture\n");

    enum { MTL_RGBA8_UNORM = 70, MTL_STORAGE_PRIVATE = 2, MTL_USAGE_SHADER_READ = 1 };
    id desc = ((id (*)(Class, SEL, unsigned long, unsigned long, unsigned long, BOOL))
               objc_msgSend)(objc_getClass("MTLTextureDescriptor"),
        sel_registerName("texture2DDescriptorWithPixelFormat:width:height:mipmapped:"),
        MTL_RGBA8_UNORM, 4, 4, NO);
    REQUIRE(desc);
    ((void (*)(id, SEL, unsigned long)) objc_msgSend)(
        desc, sel_registerName("setStorageMode:"), MTL_STORAGE_PRIVATE);
    ((void (*)(id, SEL, unsigned long)) objc_msgSend)(
        desc, sel_registerName("setUsage:"), MTL_USAGE_SHADER_READ);
    id mtex = ((id (*)(id, SEL, id)) objc_msgSend)(
        (id) metal->device, sel_registerName("newTextureWithDescriptor:"), desc);
    REQUIRE(mtex);

    pl_tex tex = pl_metal_wrap(gpu, pl_metal_wrap_params(
        .texture = (void *) mtex, .w = 4, .h = 4,
        .format = pl_find_named_fmt(gpu, "rgba8"),
    ));
    REQUIRE(tex);
    REQUIRE(tex->params.sampleable);
    REQUIRE(!tex->params.renderable && !tex->params.storable);
    REQUIRE(!tex->params.blit_dst);
    REQUIRE(!tex->params.host_readable && !tex->params.host_writable);
    pl_tex_destroy(gpu, &tex);
    ((void (*)(id, SEL)) objc_msgSend)(mtex, sel_registerName("release"));

    // pl_tex has no multisampling, so such textures must be rejected
    printf("- rejecting a multisampled texture\n");
    enum { MTL_TEX_2D_MS = 4, MTL_USAGE_RENDER_TARGET = 4 };
    ((void (*)(id, SEL, unsigned long)) objc_msgSend)(
        desc, sel_registerName("setTextureType:"), MTL_TEX_2D_MS);
    ((void (*)(id, SEL, unsigned long)) objc_msgSend)(
        desc, sel_registerName("setSampleCount:"), 4);
    ((void (*)(id, SEL, unsigned long)) objc_msgSend)(
        desc, sel_registerName("setUsage:"), MTL_USAGE_RENDER_TARGET);
    mtex = ((id (*)(id, SEL, id)) objc_msgSend)(
        (id) metal->device, sel_registerName("newTextureWithDescriptor:"), desc);
    REQUIRE(mtex);
    tex = pl_metal_wrap(gpu, pl_metal_wrap_params(
        .texture = (void *) mtex, .w = 4, .h = 4,
        .format = pl_find_named_fmt(gpu, "rgba8"),
    ));
    REQUIRE(!tex);
    ((void (*)(id, SEL)) objc_msgSend)(mtex, sel_registerName("release"));
}

// A timer reports each timed pass once, after the GPU has finished it
static void metal_timer_tests(pl_gpu gpu)
{
    printf("metal_timer_tests:\n");
    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 16, 32, PL_FMT_CAP_RENDERABLE);
    REQUIRE(fmt);
    pl_tex fbo = pl_tex_create(gpu, pl_tex_params(
        .w = 16, .h = 16, .format = fmt, .renderable = true));
    pl_timer timer = pl_timer_create(gpu);
    pl_dispatch dp = pl_dispatch_create(gpu->log, gpu);
    REQUIRE(fbo && timer && dp);

    pl_shader sh = pl_dispatch_begin(dp);
    sh_describe(sh, "solid");
    GLSL("vec4 color = vec4(1.0); \n");
    sh->output = PL_SHADER_SIG_COLOR;
    REQUIRE(pl_dispatch_finish(dp, pl_dispatch_params(
        .shader = &sh, .target = fbo, .timer = timer)));
    REQUIRE_CMP(pl_timer_query(gpu, timer), ==, 0, PRIu64); // not submitted yet
    pl_gpu_finish(gpu);
    REQUIRE_CMP(pl_timer_query(gpu, timer), >, 0, PRIu64);
    REQUIRE_CMP(pl_timer_query(gpu, timer), ==, 0, PRIu64); // reported once

    pl_dispatch_destroy(&dp);
    pl_timer_destroy(gpu, &timer);
    pl_tex_destroy(gpu, &fbo);
}

// Every format advertised as a vertex format must feed the shader the values
// its host representation encodes, in the right component order
static void metal_vertex_format_tests(pl_gpu gpu)
{
    printf("metal_vertex_format_tests:\n");
    pl_fmt fbo_fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 32, 32,
                                 PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_HOST_READABLE);
    pl_fmt pos_fmt = pl_find_vertex_fmt(gpu, PL_FMT_FLOAT, 2);
    REQUIRE(fbo_fmt && pos_fmt);
    pl_tex fbo = pl_tex_create(gpu, pl_tex_params(
        .w = 4, .h = 4, .format = fbo_fmt, .renderable = true, .host_readable = true));
    REQUIRE(fbo);

    static const uint16_t halfs[4] = { 0x3400, 0x3800, 0x3A00, 0x3C00 }; // 0.25 .. 1.0
    int tested = 0;
    for (int f = 0; f < gpu->num_formats; f++) {
        pl_fmt fmt = gpu->formats[f];
        if (!(fmt->caps & PL_FMT_CAP_VERTEX))
            continue;
        int n = fmt->num_components, bits = fmt->component_depth[0];

        // Encode component i of the host representation, and what the shader
        // should see for it
        struct { float pos[2]; uint8_t attr[16]; } verts[3] = {
            {{-1, -1}}, {{3, -1}}, {{-1, 3}},
        };
        float expect[4] = {0};
        for (int i = 0; i < n; i++) {
            REQUIRE_CMP(fmt->component_depth[i], ==, bits, "d");
            uint32_t raw;
            float val;
            switch (fmt->type) {
            case PL_FMT_UNORM: raw = (i + 1u) << (bits - 3); val = raw / (float) ((1ull << bits) - 1); break;
            case PL_FMT_SNORM: raw = (i + 1u) << (bits - 4); val = raw / (float) ((1ull << (bits - 1)) - 1); break;
            case PL_FMT_UINT:  raw = i + 1;                  val = i + 1.0f; break;
            case PL_FMT_SINT:  raw = (uint32_t) -(i + 1);     val = -(i + 1.0f); break;
            case PL_FMT_FLOAT:
                val = 0.25f * (i + 1);
                if (bits == 16) raw = halfs[i]; else memcpy(&raw, &val, sizeof(raw));
                break;
            default: pl_unreachable();
            }
            for (int v = 0; v < 3; v++)
                memcpy(&verts[v].attr[i * bits / 8], &raw, bits / 8);
            expect[fmt->sample_order[i]] = val;
        }

        bool integer = fmt->type == PL_FMT_UINT || fmt->type == PL_FMT_SINT;
        static const char *widen[] = { NULL, "float(val), 0.0, 0.0, 0.0",
            "vec2(val), 0.0, 0.0", "vec3(val), 0.0", "vec4(val)" };
        char vert[512], frag[512];
        snprintf(vert, sizeof(vert),
                 "#version 450\n"
                 "layout(location=0) in vec2 pos;\n"
                 "layout(location=1) in %s attr;\n"
                 "layout(location=0) out %s%s val;\n"
                 "void main() { gl_Position = vec4(pos, 0.0, 1.0); val = attr; }\n",
                 fmt->glsl_type, integer ? "flat " : "", fmt->glsl_type);
        snprintf(frag, sizeof(frag),
                 "#version 450\n"
                 "layout(location=0) in %s%s val;\n"
                 "layout(location=0) out vec4 out_color;\n"
                 "void main() { out_color = vec4(%s); }\n",
                 integer ? "flat " : "", fmt->glsl_type, widen[n]);

        pl_pass pass = pl_pass_create(gpu, pl_pass_params(
            .type = PL_PASS_RASTER, .target_format = fbo_fmt,
            .vertex_shader = vert, .glsl_shader = frag,
            .vertex_type = PL_PRIM_TRIANGLE_LIST,
            .vertex_stride = sizeof(verts[0]),
            .num_vertex_attribs = 2,
            .vertex_attribs = (struct pl_vertex_attrib[]) {
                { .name = "pos",  .fmt = pos_fmt, .location = 0 },
                { .name = "attr", .fmt = fmt, .location = 1,
                  .offset = offsetof(__typeof__(verts[0]), attr) },
            },
        ));
        REQUIRE(pass);
        pl_pass_run(gpu, pl_pass_run_params(
            .pass = pass, .target = fbo, .vertex_count = 3, .vertex_data = verts));
        float out[4 * 4 * 4];
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = fbo, .ptr = out)));
        for (int c = 0; c < 4; c++)
            REQUIRE_FEQ(out[c], expect[c], 1e-4);
        pl_pass_destroy(gpu, &pass);
        tested++;
    }
    printf("- %d vertex formats checked\n", tested);
    REQUIRE_CMP(tested, >, 0, "d");
    pl_tex_destroy(gpu, &fbo);
}

static void rw_store(pl_dispatch dp, pl_tex tex, bool read)
{
    pl_shader sh = pl_dispatch_begin(dp);
    REQUIRE(sh_try_compute(sh, 4, 4, false, 0));
    ident_t img = sh_desc(sh, (struct pl_shader_desc) {
        .binding.object = tex,
        .desc = {
            .name   = "img",
            .type   = PL_DESC_STORAGE_IMG,
            .access = read ? PL_DESC_ACCESS_READWRITE : PL_DESC_ACCESS_WRITEONLY,
        },
    });
    GLSL("ivec2 pos = ivec2(gl_GlobalInvocationID.xy); \n");
    if (read)
        GLSL("imageStore("$", pos, imageLoad("$", pos) * vec4(0.5)); \n", img, img);
    else
        GLSL("imageStore("$", pos, vec4(0.5)); \n", img);
    REQUIRE(pl_dispatch_compute(dp, pl_dispatch_compute_params(
        .shader = &sh, .dispatch_size = {1, 1, 1})));
}

// Every format advertising READWRITE must really work as a read_write image:
// halving a cleared 1.0 must give the same texels as storing 0.5 directly.
// dispatch.c only declares float `image2D`s, so integer formats are skipped.
static void metal_readwrite_tests(pl_gpu gpu)
{
    printf("metal_readwrite_tests:\n");
    pl_dispatch dp = pl_dispatch_create(gpu->log, gpu);
    REQUIRE(dp);
    for (int i = 0; i < gpu->num_formats; i++) {
        pl_fmt fmt = gpu->formats[i];
        if (!(fmt->caps & PL_FMT_CAP_READWRITE) ||
            fmt->type == PL_FMT_UINT || fmt->type == PL_FMT_SINT)
            continue;
        printf("- %s\n", fmt->name);
        pl_tex tex[2];
        uint8_t px[2][4 * 4 * 16];
        for (int j = 0; j < 2; j++) {
            tex[j] = pl_tex_create(gpu, pl_tex_params(
                .w = 4, .h = 4, .format = fmt, .storable = true,
                .blit_dst = true, .host_readable = true));
            REQUIRE(tex[j]);
            pl_tex_clear(gpu, tex[j], (float[4]) {1.0, 1.0, 1.0, 1.0});
            rw_store(dp, tex[j], j == 0);
            REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(
                .tex = tex[j], .ptr = px[j])));
        }
        REQUIRE_MEMEQ(px[0], px[1], 4 * 4 * fmt->texel_size);
        pl_tex_destroy(gpu, &tex[0]);
        pl_tex_destroy(gpu, &tex[1]);
    }
    pl_dispatch_destroy(&dp);
}

// A user-supplied command queue must be used, exposed and left alive
static void metal_queue_tests(pl_log log, pl_metal metal)
{
    printf("metal_queue_tests:\n");
    id queue = ((id (*)(id, SEL)) objc_msgSend)(
        (id) metal->device, sel_registerName("newCommandQueue"));
    REQUIRE(queue);
    pl_metal m2 = pl_metal_create(log, pl_metal_params(
        .device = metal->device, .queue = (void *) queue));
    REQUIRE(m2);
    REQUIRE(m2->queue == (void *) queue);
    REQUIRE(metal->queue && metal->queue != (void *) queue);

    pl_buf buf = pl_buf_create(m2->gpu, pl_buf_params(
        .size = 16, .host_writable = true, .host_readable = true));
    REQUIRE(buf);
    static const uint32_t data[4] = {1, 2, 3, 4};
    uint32_t out[4] = {0};
    pl_buf_write(m2->gpu, buf, 0, data, sizeof(data));
    REQUIRE(pl_buf_read(m2->gpu, buf, 0, out, sizeof(out)));
    REQUIRE_MEMEQ(out, data, sizeof(out));
    pl_buf_destroy(m2->gpu, &buf);
    pl_metal_destroy(&m2);

    unsigned long rc = ((unsigned long (*)(id, SEL)) objc_msgSend)(
        queue, sel_registerName("retainCount"));
    REQUIRE_CMP(rc, ==, 1, "lu");
    ((void (*)(id, SEL)) objc_msgSend)(queue, sel_registerName("release"));
}

int main(void)
{
    pl_log log = pl_test_logger();
    pl_metal metal = pl_metal_create(log, NULL);
    if (!metal) {
        pl_log_destroy(&log);
        return SKIP;
    }

    gpu_shader_tests(metal->gpu);
    gpu_interop_tests(metal->gpu);
    metal_interop_tests(metal->gpu);
    metal_iosurface_tests(metal->gpu);
    metal_swapchain_tests(metal);
    metal_swapchain_layer_tests(metal);
    metal_hdr_output_tests(metal->gpu);
    metal_swapchain_hdr_tests(metal);
    metal_download_stride_test(metal->gpu);
    metal_render_regression(metal->gpu);
    metal_hazard_tests(metal->gpu);
    metal_cache_tests(metal->gpu);
    metal_transfer_tests(metal->gpu);
    metal_wrap_tests(metal);
    metal_timer_tests(metal->gpu);
    metal_vertex_format_tests(metal->gpu);
    metal_readwrite_tests(metal->gpu);
    metal_queue_tests(log, metal);

    pl_metal_destroy(&metal);
    pl_log_destroy(&log);
}
