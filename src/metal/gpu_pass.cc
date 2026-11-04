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
 * License along with libplacebo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "gpu.h"

#include <string.h>

namespace {

static constexpr uint32_t METAL_DESC_BUF_BASE = 8;
static constexpr uint32_t METAL_PUSHC_BUFFER_SLOT = 7;

static uint64_t hash_bytes(const void *ptr, size_t size)
{
    const uint8_t *p = (const uint8_t *) ptr;
    uint64_t h = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < size; i++) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static void hash_merge_u64(uint64_t *acc, uint64_t v)
{
    *acc ^= v + UINT64_C(0x9e3779b97f4a7c15) + (*acc << 6) + (*acc >> 2);
}

struct metal_shader_bin {
    MTL::Library *library;
    MTL::Function *function;
    uint32_t wg_size[3];
};

static SpvExecutionModel glsl_stage_to_spv(enum glsl_shader_stage stage)
{
    switch (stage) {
    case GLSL_SHADER_VERTEX:   return SpvExecutionModelVertex;
    case GLSL_SHADER_FRAGMENT: return SpvExecutionModelFragment;
    case GLSL_SHADER_COMPUTE:  return SpvExecutionModelGLCompute;
    }

    return SpvExecutionModelMax;
}

static void release_shader_bin(struct metal_shader_bin *bin)
{
    if (bin->function)
        bin->function->release();
    if (bin->library)
        bin->library->release();
    *bin = {};
}

static uint32_t metal_buffer_slot_for_desc(const struct pl_desc *desc)
{
    return METAL_DESC_BUF_BASE + (uint32_t) desc->binding;
}

static bool compile_glsl_to_msl(pl_gpu gpu, enum glsl_shader_stage stage,
                                const struct pl_pass_params *params,
                                const char *glsl, struct metal_shader_bin *out)
{
    struct pl_gpu_metal *p = (struct pl_gpu_metal *) PL_PRIV(gpu);
    pl_cache gpu_cache = pl_gpu_cache(gpu);
    void *tmp = pl_tmp(NULL);
    spvc_context sc = NULL;
    spvc_parsed_ir ir = NULL;
    spvc_compiler comp = NULL;
    spvc_compiler_options opts = NULL;
    const char *msl_source = NULL;
    const spvc_entry_point *eps = NULL;
    size_t num_eps = 0;
    const char *entry_name = "main";
    SpvExecutionModel model = glsl_stage_to_spv(stage);
    pl_cache_obj cached_msl = {};
    uint64_t cache_key = UINT64_C(0x4d544c4d534c3032); // "MTLMSL02"

    hash_merge_u64(&cache_key, p->spirv->signature);
    hash_merge_u64(&cache_key, (uint64_t) stage);
    hash_merge_u64(&cache_key, hash_bytes(glsl, strlen(glsl)));
    if (params->target_format)
        hash_merge_u64(&cache_key, (uint64_t) params->target_format->signature);
    for (int i = 0; i < params->num_descriptors; i++) {
        // hash the contents, not the struct: `name` is a pointer
        const struct pl_desc *desc = &params->descriptors[i];
        hash_merge_u64(&cache_key, hash_bytes(desc->name, strlen(desc->name)));
        hash_merge_u64(&cache_key, (uint64_t) desc->type);
        hash_merge_u64(&cache_key, (uint64_t) desc->binding);
        hash_merge_u64(&cache_key, (uint64_t) desc->access);
    }

    if (gpu_cache) {
        cached_msl.key = cache_key;
        const size_t wg_bytes = sizeof(out->wg_size);
        if (pl_cache_get(gpu_cache, &cached_msl) && cached_msl.data &&
            cached_msl.size > wg_bytes + 1)
        {
            const char *cached_data = (const char *) cached_msl.data + wg_bytes;
            size_t cached_size = cached_msl.size - wg_bytes;
            size_t entry_len = strnlen(cached_data, cached_size);
            if (entry_len + 1 < cached_size) {
                const char *cached_entry = cached_data;
                const char *cached_src = cached_data + entry_len + 1;
                NS::String *src = NS::String::string(cached_src, NS::UTF8StringEncoding);
                NS::Error *err = NULL;
                MTL::Library *lib = p->device->newLibrary(src, nullptr, &err);
                if (lib) {
                    NS::String *fn_name = NS::String::string(cached_entry, NS::UTF8StringEncoding);
                    MTL::Function *func = lib->newFunction(fn_name);
                    if (func) {
                        out->library = lib;
                        out->function = func;
                        memcpy(out->wg_size, cached_msl.data, wg_bytes);
                        // pl_cache_get() hands over ownership; put it back
                        pl_cache_set(gpu_cache, &cached_msl);
                        pl_free(tmp);
                        return true;
                    }
                    lib->release();
                }
            }
            pl_cache_obj_free(&cached_msl);
        }
    }

    pl_str spirv = pl_spirv_compile_glsl(p->spirv, tmp, gpu->glsl, stage, glsl);
    if (!spirv.len)
        goto error;

#define SC(cmd)                                                             \
    do {                                                                    \
        spvc_result res = (cmd);                                            \
        if (res != SPVC_SUCCESS) {                                          \
            PL_ERR(gpu, "%s: %s (%d)", #cmd,                              \
                   sc ? spvc_context_get_last_error_string(sc) : "", res); \
            goto error;                                                     \
        }                                                                   \
    } while (0)

    SC(spvc_context_create(&sc));
    SC(spvc_context_parse_spirv(sc, (const SpvId *) spirv.buf,
                                spirv.len / sizeof(uint32_t), &ir));
    SC(spvc_context_create_compiler(sc, SPVC_BACKEND_MSL, ir,
                                    SPVC_CAPTURE_MODE_TAKE_OWNERSHIP, &comp));
    SC(spvc_compiler_create_compiler_options(comp, &opts));

    SC(spvc_compiler_options_set_uint(opts, SPVC_COMPILER_OPTION_MSL_VERSION,
                                      SPVC_MAKE_MSL_VERSION(2, 0, 0)));
    SC(spvc_compiler_options_set_bool(opts, SPVC_COMPILER_OPTION_MSL_ENABLE_DECORATION_BINDING, true));
    SC(spvc_compiler_options_set_bool(opts, SPVC_COMPILER_OPTION_MSL_TEXTURE_1D_AS_2D, true));
    if (stage == GLSL_SHADER_VERTEX) {
        SC(spvc_compiler_options_set_bool(opts, SPVC_COMPILER_OPTION_FLIP_VERTEX_Y, true));
    }
    SC(spvc_compiler_install_compiler_options(comp, opts));

    SC(spvc_compiler_get_entry_points(comp, &eps, &num_eps));
    for (size_t i = 0; i < num_eps; i++) {
        if (eps[i].execution_model == model) {
            entry_name = eps[i].name;
            break;
        }
    }

    SC(spvc_compiler_set_entry_point(comp, entry_name, model));

    for (int i = 0; i < params->num_descriptors; i++) {
        const struct pl_desc *desc = &params->descriptors[i];
        spvc_msl_resource_binding_2 rb;
        spvc_msl_resource_binding_init_2(&rb);
        rb.stage = model;
        rb.desc_set = 0;
        rb.binding = (unsigned) desc->binding;

        switch (desc->type) {
        case PL_DESC_SAMPLED_TEX:
            rb.msl_texture = (unsigned) desc->binding;
            rb.msl_sampler = (unsigned) desc->binding;
            break;
        case PL_DESC_STORAGE_IMG:
            rb.msl_texture = (unsigned) desc->binding;
            break;
        case PL_DESC_BUF_UNIFORM:
        case PL_DESC_BUF_STORAGE:
            rb.msl_buffer = metal_buffer_slot_for_desc(desc);
            break;
        case PL_DESC_BUF_TEXEL_UNIFORM:
        case PL_DESC_BUF_TEXEL_STORAGE:
            PL_ERR(gpu, "Metal texel-buffer descriptors are not implemented yet");
            goto error;
        case PL_DESC_INVALID:
        case PL_DESC_TYPE_COUNT:
            pl_unreachable();
        }

        SC(spvc_compiler_msl_add_resource_binding_2(comp, &rb));
    }

    if (params->push_constants_size) {
        spvc_msl_resource_binding_2 rb;
        spvc_msl_resource_binding_init_2(&rb);
        rb.stage = model;
        rb.desc_set = SPVC_MSL_PUSH_CONSTANT_DESC_SET;
        rb.binding = SPVC_MSL_PUSH_CONSTANT_BINDING;
        rb.msl_buffer = METAL_PUSHC_BUFFER_SLOT;
        SC(spvc_compiler_msl_add_resource_binding_2(comp, &rb));
    }

    if (stage == GLSL_SHADER_COMPUTE) {
        out->wg_size[0] = spvc_compiler_get_execution_mode_argument_by_index(comp,
            SpvExecutionModeLocalSize, 0);
        out->wg_size[1] = spvc_compiler_get_execution_mode_argument_by_index(comp,
            SpvExecutionModeLocalSize, 1);
        out->wg_size[2] = spvc_compiler_get_execution_mode_argument_by_index(comp,
            SpvExecutionModeLocalSize, 2);
        if (!out->wg_size[0]) out->wg_size[0] = 1;
        if (!out->wg_size[1]) out->wg_size[1] = 1;
        if (!out->wg_size[2]) out->wg_size[2] = 1;
    } else {
        out->wg_size[0] = 1;
        out->wg_size[1] = 1;
        out->wg_size[2] = 1;
    }

    SC(spvc_compiler_compile(comp, &msl_source));

    {
        NS::String *src = NS::String::string(msl_source, NS::UTF8StringEncoding);
        NS::Error *err = NULL;
        MTL::Library *lib = p->device->newLibrary(src, nullptr, &err);
        if (!lib) {
            const char *err_s = err && err->localizedDescription()
                ? err->localizedDescription()->utf8String() : "(unknown)";
            PL_ERR(gpu, "Metal shader compilation failed: %s", err_s);
            goto error;
        }

        const char *fn = spvc_compiler_get_cleansed_entry_point_name(comp, entry_name, model);
        NS::String *fn_name = NS::String::string(fn, NS::UTF8StringEncoding);
        MTL::Function *func = lib->newFunction(fn_name);
        if (!func) {
            lib->release();
            goto error;
        }

        out->library = lib;
        out->function = func;

        if (gpu_cache && msl_source) {
            size_t wg_bytes = sizeof(out->wg_size);
            size_t fn_len = strlen(fn) + 1;
            size_t msl_len = strlen(msl_source) + 1;
            size_t total = wg_bytes + fn_len + msl_len;
            uint8_t *cache_data = (uint8_t *) pl_alloc(NULL, total);
            memcpy(cache_data, out->wg_size, wg_bytes);
            memcpy(cache_data + wg_bytes, fn, fn_len);
            memcpy(cache_data + wg_bytes + fn_len, msl_source, msl_len);
            pl_cache_obj obj = {
                .key = cache_key,
                .data = cache_data,
                .size = total,
                .free = pl_free,
            };
            pl_cache_set(gpu_cache, &obj);
        }
    }

    pl_free(tmp);
    spvc_context_destroy(sc);
    return true;

error:
    pl_free(tmp);
    if (sc)
        spvc_context_destroy(sc);
    return false;

#undef SC
}

static MTL::PrimitiveType map_primitive(enum pl_prim_type t)
{
    switch (t) {
    case PL_PRIM_TRIANGLE_LIST:  return MTL::PrimitiveTypeTriangle;
    case PL_PRIM_TRIANGLE_STRIP: return MTL::PrimitiveTypeTriangleStrip;
    case PL_PRIM_TYPE_COUNT:     break;
    }

    return MTL::PrimitiveTypeTriangle;
}

static MTL::BlendFactor map_blend_factor(enum pl_blend_mode mode)
{
    switch (mode) {
    case PL_BLEND_ZERO:                 return MTL::BlendFactorZero;
    case PL_BLEND_ONE:                  return MTL::BlendFactorOne;
    case PL_BLEND_SRC_ALPHA:            return MTL::BlendFactorSourceAlpha;
    case PL_BLEND_ONE_MINUS_SRC_ALPHA:  return MTL::BlendFactorOneMinusSourceAlpha;
    case PL_BLEND_MODE_COUNT:           break;
    }

    return MTL::BlendFactorOne;
}

static MTL::SamplerState *get_sampler(pl_gpu gpu, enum pl_tex_sample_mode sample,
                                      enum pl_tex_address_mode address)
{
    struct pl_gpu_metal *p = (struct pl_gpu_metal *) PL_PRIV(gpu);
    MTL::SamplerState *cached = p->samplers[sample][address];
    if (cached)
        return cached;

    MTL::SamplerDescriptor *desc = MTL::SamplerDescriptor::alloc()->init();
    if (!desc)
        return NULL;

    switch (sample) {
    case PL_TEX_SAMPLE_NEAREST:
        desc->setMinFilter(MTL::SamplerMinMagFilterNearest);
        desc->setMagFilter(MTL::SamplerMinMagFilterNearest);
        desc->setMipFilter(MTL::SamplerMipFilterNearest);
        break;
    case PL_TEX_SAMPLE_LINEAR:
        desc->setMinFilter(MTL::SamplerMinMagFilterLinear);
        desc->setMagFilter(MTL::SamplerMinMagFilterLinear);
        desc->setMipFilter(MTL::SamplerMipFilterLinear);
        break;
    case PL_TEX_SAMPLE_MODE_COUNT:
        desc->release();
        return NULL;
    }

    MTL::SamplerAddressMode m = MTL::SamplerAddressModeClampToEdge;
    switch (address) {
    case PL_TEX_ADDRESS_CLAMP:  m = MTL::SamplerAddressModeClampToEdge; break;
    case PL_TEX_ADDRESS_REPEAT: m = MTL::SamplerAddressModeRepeat; break;
    case PL_TEX_ADDRESS_MIRROR: m = MTL::SamplerAddressModeMirrorRepeat; break;
    case PL_TEX_ADDRESS_MODE_COUNT:
        desc->release();
        return NULL;
    }

    desc->setSAddressMode(m);
    desc->setTAddressMode(m);
    desc->setRAddressMode(m);

    MTL::SamplerState *state = p->device->newSamplerState(desc);
    desc->release();
    if (!state)
        return NULL;

    p->samplers[sample][address] = state;
    return state;
}

static void bind_descriptors_raster(pl_gpu gpu, pl_pass pass,
                                    MTL::RenderCommandEncoder *enc,
                                    const struct pl_pass_run_params *params)
{
    for (int i = 0; i < pass->params.num_descriptors; i++) {
        const struct pl_desc *desc = &pass->params.descriptors[i];
        const struct pl_desc_binding *db = &params->desc_bindings[i];

        switch (desc->type) {
        case PL_DESC_SAMPLED_TEX: {
            pl_tex tex = (pl_tex) db->object;
            struct pl_tex_metal *tex_p = (struct pl_tex_metal *) PL_PRIV(tex);
            MTL::SamplerState *sampler = get_sampler(gpu, db->sample_mode, db->address_mode);
            if (tex_p->tex) {
                enc->setVertexTexture(tex_p->tex, (NS::UInteger) desc->binding);
                enc->setFragmentTexture(tex_p->tex, (NS::UInteger) desc->binding);
            }
            if (sampler) {
                enc->setVertexSamplerState(sampler, (NS::UInteger) desc->binding);
                enc->setFragmentSamplerState(sampler, (NS::UInteger) desc->binding);
            }
            break;
        }
        case PL_DESC_STORAGE_IMG: {
            pl_tex tex = (pl_tex) db->object;
            struct pl_tex_metal *tex_p = (struct pl_tex_metal *) PL_PRIV(tex);
            if (tex_p->tex) {
                enc->setVertexTexture(tex_p->tex, (NS::UInteger) desc->binding);
                enc->setFragmentTexture(tex_p->tex, (NS::UInteger) desc->binding);
            }
            break;
        }
        case PL_DESC_BUF_UNIFORM:
        case PL_DESC_BUF_STORAGE: {
            pl_buf buf = (pl_buf) db->object;
            struct pl_buf_metal *buf_p = (struct pl_buf_metal *) PL_PRIV(buf);
            NS::UInteger idx = (NS::UInteger) metal_buffer_slot_for_desc(desc);
            if (buf_p->buf) {
                enc->setVertexBuffer(buf_p->buf, 0, idx);
                enc->setFragmentBuffer(buf_p->buf, 0, idx);
            }
            break;
        }
        case PL_DESC_BUF_TEXEL_UNIFORM:
        case PL_DESC_BUF_TEXEL_STORAGE:
            break;
        case PL_DESC_INVALID:
        case PL_DESC_TYPE_COUNT:
            pl_unreachable();
        }
    }
}

static void bind_descriptors_compute(pl_gpu gpu, pl_pass pass, MTL::ComputeCommandEncoder *enc,
                                     const struct pl_pass_run_params *params)
{
    for (int i = 0; i < pass->params.num_descriptors; i++) {
        const struct pl_desc *desc = &pass->params.descriptors[i];
        const struct pl_desc_binding *db = &params->desc_bindings[i];

        switch (desc->type) {
        case PL_DESC_SAMPLED_TEX: {
            pl_tex tex = (pl_tex) db->object;
            struct pl_tex_metal *tex_p = (struct pl_tex_metal *) PL_PRIV(tex);
            MTL::SamplerState *sampler = get_sampler(gpu, db->sample_mode, db->address_mode);
            if (tex_p->tex)
                enc->setTexture(tex_p->tex, (NS::UInteger) desc->binding);
            if (sampler)
                enc->setSamplerState(sampler, (NS::UInteger) desc->binding);
            break;
        }
        case PL_DESC_STORAGE_IMG: {
            pl_tex tex = (pl_tex) db->object;
            struct pl_tex_metal *tex_p = (struct pl_tex_metal *) PL_PRIV(tex);
            if (tex_p->tex)
                enc->setTexture(tex_p->tex, (NS::UInteger) desc->binding);
            break;
        }
        case PL_DESC_BUF_UNIFORM:
        case PL_DESC_BUF_STORAGE: {
            pl_buf buf = (pl_buf) db->object;
            struct pl_buf_metal *buf_p = (struct pl_buf_metal *) PL_PRIV(buf);
            if (buf_p->buf)
                enc->setBuffer(buf_p->buf, 0, (NS::UInteger) metal_buffer_slot_for_desc(desc));
            break;
        }
        case PL_DESC_BUF_TEXEL_UNIFORM:
        case PL_DESC_BUF_TEXEL_STORAGE:
            break;
        case PL_DESC_INVALID:
        case PL_DESC_TYPE_COUNT:
            pl_unreachable();
        }
    }
}

static void stamp_pass_resources(pl_gpu gpu, const struct pl_pass_run_params *params)
{
    struct pl_gpu_metal *p = (struct pl_gpu_metal *) PL_PRIV(gpu);

    // Stamp target texture
    if (params->target)
        metal_stamp_tex(p, params->target);

    // Stamp vertex and index buffers
    if (params->vertex_buf)
        metal_stamp_buf(p, params->vertex_buf);
    if (params->index_buf)
        metal_stamp_buf(p, params->index_buf);

    // Stamp all descriptor bindings
    for (int i = 0; i < params->pass->params.num_descriptors; i++) {
        const struct pl_desc *desc = &params->pass->params.descriptors[i];
        const struct pl_desc_binding *db = &params->desc_bindings[i];

        switch (desc->type) {
        case PL_DESC_SAMPLED_TEX:
        case PL_DESC_STORAGE_IMG:
            metal_stamp_tex(p, (pl_tex) db->object);
            break;
        case PL_DESC_BUF_UNIFORM:
        case PL_DESC_BUF_STORAGE:
            metal_stamp_buf(p, (pl_buf) db->object);
            break;
        case PL_DESC_BUF_TEXEL_UNIFORM:
        case PL_DESC_BUF_TEXEL_STORAGE:
        case PL_DESC_INVALID:
        case PL_DESC_TYPE_COUNT:
            break;
        }
    }
}

} // namespace

struct pl_pass_metal {
    MTL::Library *vert_lib;
    MTL::Library *frag_lib;
    MTL::Library *comp_lib;
    MTL::Function *vert_fn;
    MTL::Function *frag_fn;
    MTL::Function *comp_fn;
    MTL::RenderPipelineState *raster_pso;
    MTL::ComputePipelineState *compute_pso;
    MTL::PrimitiveType primitive;
    uint32_t wg_size[3];
};

pl_pass pl_metal_pass_create(pl_gpu gpu, const struct pl_pass_params *params)
{
    metal_autorelease pool;
    struct pl_gpu_metal *p = (struct pl_gpu_metal *) PL_PRIV(gpu);

    struct pl_pass_t *pass = pl_zalloc_obj(NULL, pass, struct pl_pass_metal);
    pass->params = pl_pass_params_copy(pass, params);
    struct pl_pass_metal *pass_p = (struct pl_pass_metal *) PL_PRIV(pass);

    if (params->type == PL_PASS_RASTER) {
        struct metal_shader_bin vert = {}, frag = {};
        if (!compile_glsl_to_msl(gpu, GLSL_SHADER_VERTEX, params,
                                 params->vertex_shader, &vert))
            goto error;
        if (!compile_glsl_to_msl(gpu, GLSL_SHADER_FRAGMENT, params,
                                 params->glsl_shader, &frag)) {
            release_shader_bin(&vert);
            goto error;
        }

        pass_p->vert_lib = vert.library;
        pass_p->frag_lib = frag.library;
        pass_p->vert_fn = vert.function;
        pass_p->frag_fn = frag.function;
        pass_p->primitive = map_primitive(params->vertex_type);

        MTL::RenderPipelineDescriptor *desc = MTL::RenderPipelineDescriptor::alloc()->init();
        if (!desc) {
            PL_ERR(gpu, "Failed allocating Metal render pipeline descriptor");
            goto error;
        }

        desc->setVertexFunction(pass_p->vert_fn);
        desc->setFragmentFunction(pass_p->frag_fn);

        MTL::RenderPipelineColorAttachmentDescriptor *ca =
            desc->colorAttachments()->object((NS::UInteger) 0);
        ca->setPixelFormat(fmt_to_mtl(params->target_format));

        if (params->blend_params) {
            const struct pl_blend_params *blend = params->blend_params;
            ca->setBlendingEnabled(true);
            ca->setSourceRGBBlendFactor(map_blend_factor(blend->src_rgb));
            ca->setDestinationRGBBlendFactor(map_blend_factor(blend->dst_rgb));
            ca->setSourceAlphaBlendFactor(map_blend_factor(blend->src_alpha));
            ca->setDestinationAlphaBlendFactor(map_blend_factor(blend->dst_alpha));
            ca->setRgbBlendOperation(MTL::BlendOperationAdd);
            ca->setAlphaBlendOperation(MTL::BlendOperationAdd);
        }

        MTL::VertexDescriptor *vdesc = MTL::VertexDescriptor::alloc()->init();
        if (!vdesc) {
            desc->release();
            PL_ERR(gpu, "Failed allocating Metal vertex descriptor");
            goto error;
        }

        vdesc->layouts()->object(0)->setStride((NS::UInteger) params->vertex_stride);
        vdesc->layouts()->object(0)->setStepFunction(MTL::VertexStepFunctionPerVertex);

        for (int i = 0; i < params->num_vertex_attribs; i++) {
            const struct pl_vertex_attrib *va = &params->vertex_attribs[i];
            MTL::VertexFormat fmt = fmt_to_mtl_vertex(va->fmt);
            if (fmt == MTL::VertexFormatInvalid) {
                vdesc->release();
                desc->release();
                PL_ERR(gpu, "Unsupported Metal vertex format: %s", va->fmt->name);
                goto error;
            }

            auto attr = vdesc->attributes()->object((NS::UInteger) va->location);
            attr->setFormat(fmt);
            attr->setOffset((NS::UInteger) va->offset);
            attr->setBufferIndex(0);
        }

        desc->setVertexDescriptor(vdesc);
        vdesc->release();

        NS::Error *err = NULL;
        pass_p->raster_pso = p->device->newRenderPipelineState(desc, &err);
        desc->release();

        if (!pass_p->raster_pso) {
            const char *err_s = err && err->localizedDescription()
                ? err->localizedDescription()->utf8String() : "(unknown)";
            PL_ERR(gpu, "Failed creating Metal render pipeline state: %s", err_s);
            goto error;
        }
    } else {
        struct metal_shader_bin comp = {};
        if (!compile_glsl_to_msl(gpu, GLSL_SHADER_COMPUTE, params,
                                 params->glsl_shader, &comp))
            goto error;

        pass_p->comp_lib = comp.library;
        pass_p->comp_fn = comp.function;
        memcpy(pass_p->wg_size, comp.wg_size, sizeof(pass_p->wg_size));

        NS::Error *err = NULL;
        pass_p->compute_pso = p->device->newComputePipelineState(pass_p->comp_fn, &err);
        if (!pass_p->compute_pso) {
            const char *err_s = err && err->localizedDescription()
                ? err->localizedDescription()->utf8String() : "(unknown)";
            PL_ERR(gpu, "Failed creating Metal compute pipeline state: %s", err_s);
            goto error;
        }
    }

    return pass;

error:
    pl_metal_pass_destroy(gpu, pass);
    return NULL;
}

void pl_metal_pass_destroy(pl_gpu gpu, pl_pass pass)
{
    (void) gpu;
    struct pl_pass_metal *pass_p = (struct pl_pass_metal *) PL_PRIV(pass);

    if (pass_p->raster_pso)
        pass_p->raster_pso->release();
    if (pass_p->compute_pso)
        pass_p->compute_pso->release();

    if (pass_p->vert_fn)
        pass_p->vert_fn->release();
    if (pass_p->frag_fn)
        pass_p->frag_fn->release();
    if (pass_p->comp_fn)
        pass_p->comp_fn->release();

    if (pass_p->vert_lib)
        pass_p->vert_lib->release();
    if (pass_p->frag_lib)
        pass_p->frag_lib->release();
    if (pass_p->comp_lib)
        pass_p->comp_lib->release();

    pl_free((void *) pass);
}

static MTL::Buffer *stream_buf_upload(pl_gpu gpu,
                                       struct metal_stream_buf *stream,
                                       const void *data, size_t size,
                                       size_t *out_offset)
{
    struct pl_gpu_metal *p = (struct pl_gpu_metal *) PL_PRIV(gpu);

    // Align offset to 256 bytes (Metal buffer offset alignment requirement)
    size_t aligned_used = PL_ALIGN2(stream->used, 256);

    if (!stream->buf || aligned_used + size > stream->size) {
        // Need a new buffer; the old one (if any) is retained by any
        // in-flight command buffer encoders that reference it. Only grow
        // when it ran out, not when it was dropped on submission.
        size_t new_size = PL_MAX(16 * 1024, stream->buf ? stream->size * 2
                                                        : stream->size);
        if (stream->buf)
            stream->buf->release();

        while (new_size < size)
            new_size *= 2;

        stream->buf = p->device->newBuffer(new_size,
            MTL::ResourceStorageModeShared | MTL::ResourceCPUCacheModeWriteCombined);
        if (!stream->buf) {
            stream->size = 0;
            stream->used = 0;
            return NULL;
        }

        stream->size = new_size;
        aligned_used = 0;
    }

    uint8_t *dst = (uint8_t *) stream->buf->contents();
    if (!dst)
        return NULL;

    memcpy(dst + aligned_used, data, size);
    *out_offset = aligned_used;
    stream->used = aligned_used + size;
    return stream->buf;
}

// A dropped pass leaves its target stale, so report it as a GPU failure
static void pass_run_failed(pl_gpu gpu, const char *what)
{
    struct pl_gpu_metal *p = (struct pl_gpu_metal *) PL_PRIV(gpu);
    PL_ERR(gpu, "Failed running Metal pass: %s", what);
    p->is_failed = true;
}

void pl_metal_pass_run(pl_gpu gpu, const struct pl_pass_run_params *params)
{
    metal_autorelease pool;
    struct pl_gpu_metal *p = (struct pl_gpu_metal *) PL_PRIV(gpu);
    struct pl_pass_metal *pass_p = (struct pl_pass_metal *) PL_PRIV(params->pass);
    MTL::CommandBuffer *cb = p->pending_cb;
    if (!cb) {
        pass_run_failed(gpu, "no command buffer");
        return;
    }
    pl_metal_timer_begin(gpu, params->timer);

    if (params->pass->params.type == PL_PASS_RASTER) {
        struct pl_tex_metal *target_p = (struct pl_tex_metal *) PL_PRIV(params->target);
        if (!target_p->tex || !pass_p->raster_pso) {
            pass_run_failed(gpu, "missing target or pipeline");
            return;
        }

        MTL::RenderPassDescriptor *rp = MTL::RenderPassDescriptor::renderPassDescriptor();
        auto color = rp->colorAttachments()->object((NS::UInteger) 0);
        color->setTexture(target_p->tex);
        color->setStoreAction(MTL::StoreActionStore);
        color->setLoadAction(params->pass->params.load_target ? MTL::LoadActionLoad
                                                               : MTL::LoadActionDontCare);

        MTL::RenderCommandEncoder *enc = cb->renderCommandEncoder(rp);
        if (!enc) {
            pass_run_failed(gpu, "render encoder creation");
            return;
        }

        enc->setRenderPipelineState(pass_p->raster_pso);

        MTL::Viewport vp = {
            .originX = (double) params->viewport.x0,
            .originY = (double) params->viewport.y0,
            .width = (double) pl_rect_w(params->viewport),
            .height = (double) pl_rect_h(params->viewport),
            .znear = 0.0,
            .zfar = 1.0,
        };
        enc->setViewport(vp);

        MTL::ScissorRect sc = {
            .x = (NS::UInteger) params->scissors.x0,
            .y = (NS::UInteger) params->scissors.y0,
            .width = (NS::UInteger) pl_rect_w(params->scissors),
            .height = (NS::UInteger) pl_rect_h(params->scissors),
        };
        enc->setScissorRect(sc);

        bind_descriptors_raster(gpu, params->pass, enc, params);

        if (params->pass->params.push_constants_size && params->push_constants) {
            enc->setVertexBytes(params->push_constants,
                                (NS::UInteger) params->pass->params.push_constants_size,
                                METAL_PUSHC_BUFFER_SLOT);
            enc->setFragmentBytes(params->push_constants,
                                  (NS::UInteger) params->pass->params.push_constants_size,
                                  METAL_PUSHC_BUFFER_SLOT);
        }

        if (params->vertex_buf) {
            struct pl_buf_metal *vb = (struct pl_buf_metal *) PL_PRIV(params->vertex_buf);
            enc->setVertexBuffer(vb->buf, (NS::UInteger) params->buf_offset, 0);
        } else {
            size_t vbo_size = pl_vertex_buf_size(params);
            size_t vbo_offset = 0;
            MTL::Buffer *vbo = stream_buf_upload(gpu, &p->vbuf, params->vertex_data,
                                                  vbo_size, &vbo_offset);
            if (!vbo) {
                enc->endEncoding();
                pass_run_failed(gpu, "vertex buffer allocation");
                return;
            }
            enc->setVertexBuffer(vbo, (NS::UInteger) vbo_offset, 0);
        }

        if (params->index_data || params->index_buf) {
            MTL::Buffer *idx_buf = NULL;
            size_t idx_off = 0;
            MTL::IndexType idx_ty = params->index_fmt == PL_INDEX_UINT16
                ? MTL::IndexTypeUInt16 : MTL::IndexTypeUInt32;

            if (params->index_buf) {
                struct pl_buf_metal *ib = (struct pl_buf_metal *) PL_PRIV(params->index_buf);
                idx_buf = ib->buf;
                idx_off = params->index_offset;
            } else {
                size_t idx_size = pl_index_buf_size(params);
                idx_buf = stream_buf_upload(gpu, &p->ibuf, params->index_data,
                                             idx_size, &idx_off);
                if (!idx_buf) {
                    enc->endEncoding();
                    pass_run_failed(gpu, "index buffer allocation");
                    return;
                }
            }

            enc->drawIndexedPrimitives(pass_p->primitive, (NS::UInteger) params->vertex_count,
                                       idx_ty, idx_buf, (NS::UInteger) idx_off);
        } else {
            enc->drawPrimitives(pass_p->primitive,
                                (NS::UInteger) 0,
                                (NS::UInteger) params->vertex_count);
        }

        enc->endEncoding();
        stamp_pass_resources(gpu, params);
        return;
    }

    if (!pass_p->compute_pso) {
        pass_run_failed(gpu, "missing pipeline");
        return;
    }

    MTL::ComputeCommandEncoder *enc = cb->computeCommandEncoder();
    if (!enc) {
        pass_run_failed(gpu, "compute encoder creation");
        return;
    }

    enc->setComputePipelineState(pass_p->compute_pso);
    bind_descriptors_compute(gpu, params->pass, enc, params);

    if (params->pass->params.push_constants_size && params->push_constants) {
        enc->setBytes(params->push_constants,
                      (NS::UInteger) params->pass->params.push_constants_size,
                      METAL_PUSHC_BUFFER_SLOT);
    }

    MTL::Size tg = MTL::Size(pass_p->wg_size[0], pass_p->wg_size[1], pass_p->wg_size[2]);
    MTL::Size ng = MTL::Size(params->compute_groups[0],
                             params->compute_groups[1],
                             params->compute_groups[2]);
    enc->dispatchThreadgroups(ng, tg);
    enc->endEncoding();
    stamp_pass_resources(gpu, params);
}
