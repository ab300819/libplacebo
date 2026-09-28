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

static inline uint8_t *metal_buf_contents(pl_gpu gpu, pl_buf buf)
{
    struct pl_buf_metal *buf_p = (struct pl_buf_metal *) PL_PRIV(buf);
    void *ptr = buf_p->buf ? buf_p->buf->contents() : NULL;
    if (!ptr)
        PL_ERR(gpu, "Failed retrieving Metal buffer contents");
    return (uint8_t *) ptr;
}

pl_buf pl_metal_buf_create(pl_gpu gpu, const struct pl_buf_params *params)
{
    struct pl_gpu_metal *p = (struct pl_gpu_metal *) PL_PRIV(gpu);

    struct pl_buf_t *buf = pl_zalloc_obj(NULL, buf, struct pl_buf_metal);
    buf->params = *params;
    buf->params.initial_data = NULL;

    MTL::ResourceOptions opts = MTL::ResourceStorageModeShared;
    opts |= params->host_readable ? MTL::ResourceCPUCacheModeDefaultCache
                                  : MTL::ResourceCPUCacheModeWriteCombined;

    size_t size = PL_ALIGN2(params->size, 4);

    struct pl_buf_metal *buf_p = (struct pl_buf_metal *) PL_PRIV(buf);
    buf_p->buf = p->device->newBuffer(size, opts);
    if (!buf_p->buf) {
        PL_ERR(gpu, "Failed creating Metal buffer of size %zu", params->size);
        goto error;
    }

    if (params->host_mapped)
        buf->data = (uint8_t *) buf_p->buf->contents();

    if (params->initial_data)
        pl_metal_buf_write(gpu, buf, 0, params->initial_data, params->size);

    return buf;

error:
    pl_metal_buf_destroy(gpu, buf);
    return NULL;
}

void pl_metal_buf_destroy(pl_gpu gpu, pl_buf buf)
{
    (void) gpu;

    struct pl_buf_metal *buf_p = (struct pl_buf_metal *) PL_PRIV(buf);
    if (buf_p->buf)
        buf_p->buf->release();

    pl_free((void *) buf);
}

void pl_metal_buf_write(pl_gpu gpu, pl_buf buf, size_t offset, const void *data,
                        size_t size)
{
    struct pl_gpu_metal *p = (struct pl_gpu_metal *) PL_PRIV(gpu);
    metal_wait_gen(p, ((struct pl_buf_metal *) PL_PRIV(buf))->last_gen, UINT64_MAX);

    uint8_t *dst = metal_buf_contents(gpu, buf);
    if (!dst)
        return;

    memcpy(dst + offset, data, size);
}

bool pl_metal_buf_read(pl_gpu gpu, pl_buf buf, size_t offset, void *dest,
                       size_t size)
{
    struct pl_gpu_metal *p = (struct pl_gpu_metal *) PL_PRIV(gpu);
    struct pl_buf_metal *buf_p = (struct pl_buf_metal *) PL_PRIV(buf);

    // Wait only for this buffer's last GPU usage instead of a full finish
    if (!metal_wait_gen(p, buf_p->last_gen, UINT64_MAX))
        return false;

    const uint8_t *src = metal_buf_contents(gpu, buf);
    if (!src)
        return false;

    memcpy(dest, src + offset, size);
    return true;
}

void pl_metal_buf_copy(pl_gpu gpu, pl_buf dst, size_t dst_offset, pl_buf src,
                       size_t src_offset, size_t size)
{
    struct pl_gpu_metal *p = (struct pl_gpu_metal *) PL_PRIV(gpu);
    struct pl_buf_metal *dst_p = (struct pl_buf_metal *) PL_PRIV(dst);
    struct pl_buf_metal *src_p = (struct pl_buf_metal *) PL_PRIV(src);

    metal_autorelease pool;
    if (!dst_p->buf || !src_p->buf || !p->pending_cb)
        return;

    MTL::BlitCommandEncoder *blit = p->pending_cb->blitCommandEncoder();
    if (!blit)
        return;

    blit->copyFromBuffer(src_p->buf, (NS::UInteger) src_offset,
                         dst_p->buf, (NS::UInteger) dst_offset,
                         (NS::UInteger) size);
    blit->endEncoding();

    metal_stamp_buf(p, src);
    metal_stamp_buf(p, dst);
}

bool pl_metal_buf_poll(pl_gpu gpu, pl_buf buf, uint64_t timeout)
{
    struct pl_gpu_metal *p = (struct pl_gpu_metal *) PL_PRIV(gpu);
    struct pl_buf_metal *buf_p = (struct pl_buf_metal *) PL_PRIV(buf);

    return !metal_wait_gen(p, buf_p->last_gen, timeout);
}
