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

#ifdef __APPLE__
#include <IOSurface/IOSurfaceRef.h>
#endif

// `rc` has been normalized by the caller (see fix_tex_transfer in gpu.c)
static MTL::Region mtl_region(pl_rect3d rc)
{
    return MTL::Region::Make3D((NS::UInteger) rc.x0, (NS::UInteger) rc.y0,
                               (NS::UInteger) rc.z0, (NS::UInteger) pl_rect_w(rc),
                               (NS::UInteger) pl_rect_h(rc), (NS::UInteger) pl_rect_d(rc));
}

// replaceRegion takes the caller's row and slice pitch as is. 1D textures are
// created as 2D (height=1) to match MSL_TEXTURE_1D_AS_2D shader code.
static bool upload_to_metal(pl_gpu gpu, pl_tex tex, pl_rect3d rc,
                            const void *src, size_t row_pitch, size_t depth_pitch)
{
    struct pl_gpu_metal *p = (struct pl_gpu_metal *) PL_PRIV(gpu);
    struct pl_tex_metal *tex_p = (struct pl_tex_metal *) PL_PRIV(tex);
    if (!tex_p->tex)
        return false;

    // replaceRegion writes immediately, so wait out queued GPU use first.
    // TODO: upload through a staging buffer and blit, to avoid the stall
    metal_wait_gen(p, tex_p->last_gen, UINT64_MAX);

    if (pl_tex_params_dimension(tex->params) <= 2) {
        tex_p->tex->replaceRegion(mtl_region(rc), 0, src, row_pitch);
    } else {
        tex_p->tex->replaceRegion(mtl_region(rc), 0, 0, src, row_pitch, depth_pitch);
    }
    return true;
}

pl_tex pl_metal_tex_create(pl_gpu gpu, const struct pl_tex_params *params)
{
    struct pl_gpu_metal *p = (struct pl_gpu_metal *) PL_PRIV(gpu);
    MTL::TextureDescriptor *desc = NULL;
    MTL::TextureUsage usage = MTL::TextureUsageUnknown;

    struct pl_tex_t *tex = pl_zalloc_obj(NULL, tex, struct pl_tex_metal);
    tex->sampler_type = PL_SAMPLER_NORMAL;
    tex->params = *params;
    tex->params.initial_data = NULL;

    struct pl_tex_metal *tex_p = (struct pl_tex_metal *) PL_PRIV(tex);

    if (params->import_handle == PL_HANDLE_MTL_TEX) {
        tex_p->tex = (MTL::Texture *) params->shared_mem.handle.handle;
        if (!tex_p->tex) {
            PL_ERR(gpu, "Missing MTLTexture handle for PL_HANDLE_MTL_TEX import");
            goto error;
        }

        // Shader-based blits sample the source and render to the destination
        // (see below); an imported texture gets that only if its usage allows
        MTL::TextureUsage imported = tex_p->tex->usage();
        if (params->blit_src && (imported & MTL::TextureUsageShaderRead))
            tex->params.sampleable = true;
        if (params->blit_dst && (imported & MTL::TextureUsageRenderTarget))
            tex->params.renderable = true;

        tex_p->tex->retain();
        tex->shared_mem = params->shared_mem;
        return tex;
    }

    if (params->import_handle == PL_HANDLE_IOSURFACE) {
#ifdef __APPLE__
        IOSurfaceRef io = (IOSurfaceRef) params->shared_mem.handle.handle;
        if (!io) {
            PL_ERR(gpu, "Missing IOSurface handle for PL_HANDLE_IOSURFACE import");
            goto error;
        }

        desc = MTL::TextureDescriptor::alloc()->init();
        if (!desc) {
            PL_ERR(gpu, "Failed allocating Metal texture descriptor for IOSurface");
            goto error;
        }

        desc->setPixelFormat(fmt_to_mtl(params->format));
        desc->setWidth((NS::UInteger) params->w);
        desc->setHeight((NS::UInteger) PL_DEF(params->h, 1));
        desc->setMipmapLevelCount(1);
        desc->setSampleCount(1);
        desc->setTextureType(MTL::TextureType2D);
        desc->setStorageMode(MTL::StorageModeShared);
        desc->setUsage((MTL::TextureUsage) (MTL::TextureUsageShaderRead |
                                            MTL::TextureUsageShaderWrite |
                                            MTL::TextureUsageRenderTarget));

        tex_p->tex = p->device->newTexture(desc, io,
                    (NS::UInteger) params->shared_mem.plane);
        desc->release();
        desc = NULL;
        if (!tex_p->tex) {
            PL_ERR(gpu, "Failed creating Metal texture from IOSurface");
            goto error;
        }

        tex->shared_mem = params->shared_mem;
        return tex;
#else
        PL_ERR(gpu, "IOSurface import requested on non-Apple target");
        goto error;
#endif
    }

    desc = MTL::TextureDescriptor::alloc()->init();
    if (!desc) {
        PL_ERR(gpu, "Failed allocating Metal texture descriptor");
        goto error;
    }

    desc->setPixelFormat(fmt_to_mtl(params->format));
    desc->setWidth((NS::UInteger) params->w);
    desc->setHeight((NS::UInteger) PL_DEF(params->h, 1));
    desc->setDepth((NS::UInteger) PL_DEF(params->d, 1));
    desc->setMipmapLevelCount(1);
    desc->setSampleCount(1);
    // Use StorageModePrivate for GPU-only textures: they benefit from GPU
    // lossless compression, significantly reducing memory footprint. Exclude
    // textures with initial_data because Private textures do not support
    // replaceRegion, which is needed for the initial upload.
    if (!params->host_writable && !params->host_readable &&
        !params->export_handle && !params->initial_data)
    {
        desc->setStorageMode(MTL::StorageModePrivate);
    } else {
        desc->setStorageMode(MTL::StorageModeShared);
    }

    switch (pl_tex_params_dimension(*params)) {
    case 1: // Fall through: create as 2D (height=1) because SPIRV-Cross
            // MSL_TEXTURE_1D_AS_2D converts shader sampler1D to texture2d
    case 2: desc->setTextureType(MTL::TextureType2D); break;
    case 3: desc->setTextureType(MTL::TextureType3D); break;
    default:
        desc->release();
        goto error;
    }

    // Scaled, flipped or converting blits and clears use a render pass, as
    // in d3d11 below FL11_0
    tex->params.sampleable |= params->blit_src;
    tex->params.renderable |= params->blit_dst;

    if (tex->params.sampleable)
        usage = (MTL::TextureUsage) (usage | MTL::TextureUsageShaderRead);
    if (params->storable)
        usage = (MTL::TextureUsage) (usage | MTL::TextureUsageShaderWrite);
    if (tex->params.renderable)
        usage = (MTL::TextureUsage) (usage | MTL::TextureUsageRenderTarget);
    desc->setUsage(usage);

    tex_p->tex = p->device->newTexture(desc);
    desc->release();

    if (!tex_p->tex) {
        PL_ERR(gpu, "Failed creating Metal texture (%dx%dx%d, fmt %s)",
               params->w, params->h, params->d, params->format->name);
        goto error;
    }

    if (params->export_handle == PL_HANDLE_MTL_TEX) {
        tex_p->tex->retain(); // extra retain for the exported handle
        tex->shared_mem = (struct pl_shared_mem) {
            .handle = { .handle = (void *) tex_p->tex },
            .size = (size_t) params->w * PL_DEF(params->h, 1) *
                    PL_DEF(params->d, 1) * params->format->texel_size,
        };
    }

    if (params->initial_data) {
        pl_rect3d rc = { 0, 0, 0, params->w, PL_DEF(params->h, 1), PL_DEF(params->d, 1) };
        size_t row_pitch = (size_t) params->w * params->format->texel_size;
        if (!upload_to_metal(gpu, tex, rc, params->initial_data, row_pitch,
                             row_pitch * PL_DEF(params->h, 1)))
            goto error;
    }

    return tex;

error:
    pl_metal_tex_destroy(gpu, tex);
    return NULL;
}

void pl_metal_tex_destroy(pl_gpu gpu, pl_tex tex)
{
    (void) gpu;
    struct pl_tex_metal *tex_p = (struct pl_tex_metal *) PL_PRIV(tex);
    if (tex_p->tex) {
        tex_p->tex->release();
        // Release the extra retain from export_handle
        if (tex->params.export_handle == PL_HANDLE_MTL_TEX)
            tex_p->tex->release();
    }

    pl_free((void *) tex);
}

bool pl_metal_tex_upload(pl_gpu gpu, const struct pl_tex_transfer_params *params)
{
    const uint8_t *src = (const uint8_t *) params->ptr;
    if (params->buf) {
        struct pl_buf_metal *buf_p = (struct pl_buf_metal *) PL_PRIV(params->buf);
        metal_wait_gen((struct pl_gpu_metal *) PL_PRIV(gpu), buf_p->last_gen, UINT64_MAX);
        void *ptr = buf_p->buf ? buf_p->buf->contents() : NULL;
        if (!ptr) {
            PL_ERR(gpu, "Failed mapping Metal upload buffer");
            return false;
        }
        src = (const uint8_t *) ptr + params->buf_offset;
    }

    return upload_to_metal(gpu, params->tex, params->rc,
                           src, params->row_pitch, params->depth_pitch);
}

bool pl_metal_tex_download(pl_gpu gpu, const struct pl_tex_transfer_params *params)
{
    struct pl_gpu_metal *p = (struct pl_gpu_metal *) PL_PRIV(gpu);
    struct pl_tex_metal *tex_p = (struct pl_tex_metal *) PL_PRIV(params->tex);

    // Wait only for this texture's last GPU usage instead of a full finish
    if (!metal_wait_gen(p, tex_p->last_gen, UINT64_MAX))
        return false;

    uint8_t *dst = (uint8_t *) params->ptr;
    if (params->buf) {
        struct pl_buf_metal *buf_p = (struct pl_buf_metal *) PL_PRIV(params->buf);
        metal_wait_gen(p, buf_p->last_gen, UINT64_MAX);
        void *ptr = buf_p->buf ? buf_p->buf->contents() : NULL;
        if (!ptr) {
            PL_ERR(gpu, "Failed mapping Metal download buffer");
            return false;
        }
        dst = (uint8_t *) ptr + params->buf_offset;
    }

    if (!tex_p->tex)
        return false;

    if (pl_tex_params_dimension(params->tex->params) <= 2) {
        tex_p->tex->getBytes(dst, params->row_pitch, mtl_region(params->rc), 0);
    } else {
        tex_p->tex->getBytes(dst, params->row_pitch, params->depth_pitch,
                             mtl_region(params->rc), 0, 0);
    }

    return true;
}

void pl_metal_tex_clear_ex(pl_gpu gpu, pl_tex tex, const union pl_clear_color color)
{
    metal_autorelease pool;
    struct pl_gpu_metal *p = (struct pl_gpu_metal *) PL_PRIV(gpu);
    struct pl_tex_metal *tex_p = (struct pl_tex_metal *) PL_PRIV(tex);
    if (!tex_p->tex || !p->pending_cb)
        return;

    // blit_dst implies renderable (see pl_metal_tex_create), so a render pass
    // clear always works and handles every renderable format natively
    MTL::RenderPassDescriptor *rp = MTL::RenderPassDescriptor::renderPassDescriptor();
    auto ca = rp->colorAttachments()->object((NS::UInteger) 0);
    ca->setTexture(tex_p->tex);
    ca->setLoadAction(MTL::LoadActionClear);
    ca->setStoreAction(MTL::StoreActionStore);
    ca->setClearColor(MTL::ClearColor(color.f[0], color.f[1], color.f[2], color.f[3]));
    MTL::RenderCommandEncoder *enc = p->pending_cb->renderCommandEncoder(rp);
    if (enc)
        enc->endEncoding();
    metal_stamp_tex(p, tex);
}

void pl_metal_tex_blit(pl_gpu gpu, const struct pl_tex_blit_params *params)
{
    struct pl_gpu_metal *p = (struct pl_gpu_metal *) PL_PRIV(gpu);
    pl_tex src = params->src;
    pl_tex dst = params->dst;
    struct pl_tex_metal *src_p = (struct pl_tex_metal *) PL_PRIV(src);
    struct pl_tex_metal *dst_p = (struct pl_tex_metal *) PL_PRIV(dst);
    metal_autorelease pool;
    if (!src_p->tex || !dst_p->tex || !p->pending_cb)
        return;

    // The blit encoder can only copy, so anything that scales, flips or
    // converts goes through a shader instead (as in d3d11)
    pl_rect3d src_rc = params->src_rc, dst_rc = params->dst_rc;
    if (src->params.format != dst->params.format ||
        pl_rect_w(src_rc) != pl_rect_w(dst_rc) ||
        pl_rect_h(src_rc) != pl_rect_h(dst_rc) ||
        pl_rect_d(src_rc) != pl_rect_d(dst_rc))
    {
        if (!pl_tex_blit_compute(gpu, params))
            pl_tex_blit_raster(gpu, params);
        return;
    }

    MTL::BlitCommandEncoder *blit = p->pending_cb->blitCommandEncoder();
    if (!blit)
        return;

    // Equal signed extents: any flip is shared by both sides and cancels out
    pl_rect3d_normalize(&src_rc);
    pl_rect3d_normalize(&dst_rc);
    blit->copyFromTexture(
        src_p->tex, 0, 0,
        MTL::Origin((NS::UInteger) src_rc.x0, (NS::UInteger) src_rc.y0,
                    (NS::UInteger) src_rc.z0),
        MTL::Size((NS::UInteger) pl_rect_w(src_rc), (NS::UInteger) pl_rect_h(src_rc),
                  (NS::UInteger) pl_rect_d(src_rc)),
        dst_p->tex, 0, 0,
        MTL::Origin((NS::UInteger) dst_rc.x0, (NS::UInteger) dst_rc.y0,
                    (NS::UInteger) dst_rc.z0));
    blit->endEncoding();
    metal_stamp_tex(p, src);
    metal_stamp_tex(p, dst);
}

bool pl_metal_tex_poll(pl_gpu gpu, pl_tex tex, uint64_t timeout)
{
    struct pl_gpu_metal *p = (struct pl_gpu_metal *) PL_PRIV(gpu);
    struct pl_tex_metal *tex_p = (struct pl_tex_metal *) PL_PRIV(tex);

    return !metal_wait_gen(p, tex_p->last_gen, timeout);
}

extern "C"
pl_tex pl_metal_wrap(pl_gpu gpu, const struct pl_metal_wrap_params *params)
{
    if (!gpu || !params || !params->texture || !params->format || params->w <= 0 ||
        params->h <= 0)
    {
        return NULL;
    }

    // Only claim what both the format and the texture itself allow. Blits
    // may sample the source and render to the destination.
    MTL::Texture *mtex = (MTL::Texture *) params->texture;
    enum pl_fmt_caps caps = params->format->caps;
    MTL::TextureUsage usage = mtex->usage();
    bool read = usage & MTL::TextureUsageShaderRead;
    bool write = usage & MTL::TextureUsageShaderWrite;
    bool render = usage & MTL::TextureUsageRenderTarget;
    bool host = mtex->storageMode() == MTL::StorageModeShared ||
                mtex->storageMode() == MTL::StorageModeManaged;

    struct pl_tex_params tp = {
        .w = params->w,
        .h = params->h,
        .format = params->format,
        .sampleable = read && (caps & PL_FMT_CAP_SAMPLEABLE),
        .renderable = render && (caps & PL_FMT_CAP_RENDERABLE),
        .storable = write && (caps & PL_FMT_CAP_STORABLE),
        .blit_src = read && (caps & PL_FMT_CAP_BLITTABLE),
        .blit_dst = render && (caps & PL_FMT_CAP_BLITTABLE),
        .host_writable = host,
        .host_readable = host && (caps & PL_FMT_CAP_HOST_READABLE),
        .import_handle = PL_HANDLE_MTL_TEX,
        .shared_mem = {
            .handle = { .handle = params->texture },
            .plane = 0,
        },
        .debug_tag = params->debug_tag,
    };

    return pl_tex_create(gpu, &tp);
}
