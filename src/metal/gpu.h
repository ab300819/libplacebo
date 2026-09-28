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

#pragma once

#include <Metal/Metal.hpp>
#include <atomic>
#include <memory>
#include <spirv_cross_c.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "../gpu.h"
#include "../glsl/spirv.h"

#ifdef __cplusplus
}
#endif

#include "common.h"
#include "formats.h"

#ifdef __cplusplus
extern "C"
#endif
pl_gpu pl_gpu_create_metal(struct metal_ctx *ctx);

// --- pl_gpu internal structs

struct metal_stream_buf {
    MTL::Buffer *buf;
    size_t size;
    size_t used;
};

// pl_gpu_fns impl must be first member (src/gpu.h:32-33)
struct pl_gpu_metal {
    struct pl_gpu_fns impl;
    struct metal_ctx *ctx;
    MTL::Device *device;
    MTL::CommandQueue *queue;
    pl_spirv spirv;

    // Synchronization
    MTL::CommandBuffer *pending_cb;
    MTL::SharedEvent *sync_event;
    uint64_t current_gen;       // generation of pending_cb
    bool is_failed;
    // Set by command buffer completion handlers, which run on another thread
    // and may outlive this pl_gpu (see metal_advance_cmd_buffer)
    std::shared_ptr<std::atomic<bool>> *cb_failed;

    // Streaming VBO/IBO ring buffers
    struct metal_stream_buf vbuf, ibuf;

    // Pre-created samplers (same pattern as D3D11 gpu.h:85)
    MTL::SamplerState *samplers[PL_TEX_SAMPLE_MODE_COUNT][PL_TEX_ADDRESS_MODE_COUNT];
};

struct pl_buf_metal {
    MTL::Buffer *buf;
    uint64_t last_gen;
};

struct pl_tex_metal {
    MTL::Texture *tex;
    uint64_t last_gen;
};

struct pl_pass_metal;

// Per-resource generation tracking helpers
static inline void metal_stamp_buf(struct pl_gpu_metal *p, pl_buf buf) {
    ((struct pl_buf_metal *) PL_PRIV(buf))->last_gen = p->current_gen;
}
static inline void metal_stamp_tex(struct pl_gpu_metal *p, pl_tex tex) {
    ((struct pl_tex_metal *) PL_PRIV(tex))->last_gen = p->current_gen;
}
static inline bool metal_is_failed(struct pl_gpu_metal *p) {
    return p->is_failed || (p->cb_failed && (*p->cb_failed)->load());
}
static inline bool metal_gen_completed(struct pl_gpu_metal *p, uint64_t gen) {
    return gen == 0 || p->sync_event->signaledValue() >= gen;
}

void pl_metal_buf_destroy(pl_gpu gpu, pl_buf buf);
pl_buf pl_metal_buf_create(pl_gpu gpu, const struct pl_buf_params *params);
void pl_metal_buf_write(pl_gpu gpu, pl_buf buf, size_t offset, const void *data,
                        size_t size);
bool pl_metal_buf_read(pl_gpu gpu, pl_buf buf, size_t offset, void *dest,
                       size_t size);
void pl_metal_buf_copy(pl_gpu gpu, pl_buf dst, size_t dst_offset, pl_buf src,
                       size_t src_offset, size_t size);
bool pl_metal_buf_poll(pl_gpu gpu, pl_buf buf, uint64_t timeout);

void pl_metal_tex_destroy(pl_gpu gpu, pl_tex tex);
pl_tex pl_metal_tex_create(pl_gpu gpu, const struct pl_tex_params *params);
void pl_metal_tex_clear_ex(pl_gpu gpu, pl_tex tex, const union pl_clear_color color);
void pl_metal_tex_blit(pl_gpu gpu, const struct pl_tex_blit_params *params);
bool pl_metal_tex_upload(pl_gpu gpu, const struct pl_tex_transfer_params *params);
bool pl_metal_tex_download(pl_gpu gpu, const struct pl_tex_transfer_params *params);
bool pl_metal_tex_poll(pl_gpu gpu, pl_tex tex, uint64_t timeout);

void pl_metal_pass_destroy(pl_gpu gpu, pl_pass pass);
pl_pass pl_metal_pass_create(pl_gpu gpu, const struct pl_pass_params *params);
void pl_metal_pass_run(pl_gpu gpu, const struct pl_pass_run_params *params);

void pl_metal_timer_begin(pl_gpu gpu, pl_timer timer);

// Drains the autoreleased Metal objects (encoders, descriptors, command
// buffers) created within its scope. Callers may have no pool of their own,
// e.g. headless use outside of a swapchain frame.
struct metal_autorelease {
    NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
    ~metal_autorelease() { pool->release(); }
};

// Commit the pending command buffer (optionally wait) and create a new one
bool metal_advance_cmd_buffer(struct pl_gpu_metal *p, bool wait);

// Wait up to `timeout` nanoseconds for the GPU to finish generation `gen`,
// submitting it first if it is still pending. Returns whether it finished.
bool metal_wait_gen(struct pl_gpu_metal *p, uint64_t gen, uint64_t timeout);
