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

struct pl_timer_t {
    MTL::CommandBuffer *timed_cb; // retained until its GPU time is read
};

bool metal_advance_cmd_buffer(struct pl_gpu_metal *p, bool wait)
{
    if (!p->pending_cb)
        return true;

    metal_autorelease pool;

    // Signal the sync event with this CB's generation before committing
    if (p->sync_event)
        p->pending_cb->encodeSignalEvent(p->sync_event, p->current_gen);

    // Execution errors of asynchronous submissions only surface after we have
    // moved on, so record them from the completion handler. It only touches
    // the shared flag, since it may run after this pl_gpu is gone.
    std::shared_ptr<std::atomic<bool>> failed = *p->cb_failed;
    p->pending_cb->addCompletedHandler([failed](MTL::CommandBuffer *cb) {
        if (cb->status() == MTL::CommandBufferStatusError)
            failed->store(true);
    });

    p->pending_cb->commit();
    if (wait)
        p->pending_cb->waitUntilCompleted();

    if (p->pending_cb->status() == MTL::CommandBufferStatusError)
        p->is_failed = true;

    MTL::CommandBuffer *next = p->queue->commandBuffer();
    if (!next) {
        p->is_failed = true;
        p->pending_cb->release();
        p->pending_cb = NULL;
        return false;
    }

    next->retain();
    p->pending_cb->release();
    p->pending_cb = next;
    p->current_gen++;

    // The streaming buffers may still be read by the command buffer just
    // committed, which retains them. Only reuse them once it has completed;
    // otherwise drop them and let the next upload allocate fresh ones.
    // TODO: recycle them by generation instead of allocating per submission
    for (struct metal_stream_buf *s : { &p->vbuf, &p->ibuf }) {
        if (!wait && s->buf) {
            s->buf->release();
            s->buf = NULL;
        }
        s->used = 0;
    }

    return true;
}

bool metal_wait_gen(struct pl_gpu_metal *p, uint64_t gen, uint64_t timeout)
{
    if (metal_gen_completed(p, gen))
        return true;
    if (gen >= p->current_gen)
        metal_advance_cmd_buffer(p, false);

    // A failed command buffer never reaches its signal, so wait in slices
    // and give up once a failure has been recorded
    uint64_t left = timeout == UINT64_MAX ? UINT64_MAX
                                          : timeout / 1000000 + !!(timeout % 1000000);
    while (left && !metal_gen_completed(p, gen) && !metal_is_failed(p)) {
        uint64_t slice = PL_MIN(left, 100);
        p->sync_event->waitUntilSignaledValue(gen, slice);
        if (left != UINT64_MAX)
            left -= slice;
    }

    if (!metal_gen_completed(p, gen) && metal_is_failed(p)) {
        PL_ERR(p->ctx, "Metal command buffer failed, abandoning wait");
        return false;
    }
    return metal_gen_completed(p, gen);
}

static int metal_desc_namespace(pl_gpu gpu, enum pl_desc_type type)
{
    (void) gpu;
    (void) type;
    return 0;
}

static void metal_gpu_flush(pl_gpu gpu)
{
    struct pl_gpu_metal *p = (struct pl_gpu_metal *) PL_PRIV(gpu);
    if (!p->pending_cb)
        return;

    metal_advance_cmd_buffer(p, false);
}

static void metal_gpu_finish(pl_gpu gpu)
{
    struct pl_gpu_metal *p = (struct pl_gpu_metal *) PL_PRIV(gpu);
    if (!p->pending_cb)
        return;

    metal_advance_cmd_buffer(p, true);
}

static bool metal_gpu_is_failed(pl_gpu gpu)
{
    struct pl_gpu_metal *p = (struct pl_gpu_metal *) PL_PRIV(gpu);
    return metal_is_failed(p);
}

static pl_timer metal_timer_create(pl_gpu gpu)
{
    (void) gpu;
    struct pl_timer_t *timer = pl_alloc_ptr(NULL, timer);
    *timer = {};
    return timer;
}

static void metal_timer_destroy(pl_gpu gpu, pl_timer timer)
{
    (void) gpu;
    if (timer->timed_cb)
        timer->timed_cb->release();
    pl_free(timer);
}

// Reports each timed command buffer once, as soon as the GPU has finished it
static uint64_t metal_timer_query(pl_gpu gpu, pl_timer timer)
{
    (void) gpu;
    MTL::CommandBuffer *cb = timer->timed_cb;
    if (!cb || cb->status() < MTL::CommandBufferStatusCompleted)
        return 0;

    CFTimeInterval elapsed = cb->GPUEndTime() - cb->GPUStartTime();
    cb->release();
    timer->timed_cb = NULL;
    return elapsed > 0 ? (uint64_t) (elapsed * 1e9) : 1;
}

void pl_metal_timer_begin(pl_gpu gpu, pl_timer timer)
{
    struct pl_gpu_metal *p = (struct pl_gpu_metal *) PL_PRIV(gpu);
    if (!timer || !p->pending_cb)
        return;

    // Release any previous timed CB that wasn't collected
    if (timer->timed_cb)
        timer->timed_cb->release();

    // Retain the current pending command buffer for GPU time readback
    timer->timed_cb = p->pending_cb;
    timer->timed_cb->retain();
}

static void metal_gpu_destroy(pl_gpu gpu)
{
    struct pl_gpu_metal *p = (struct pl_gpu_metal *) PL_PRIV(gpu);

    for (int s = 0; s < PL_TEX_SAMPLE_MODE_COUNT; s++) {
        for (int a = 0; a < PL_TEX_ADDRESS_MODE_COUNT; a++) {
            if (p->samplers[s][a])
                p->samplers[s][a]->release();
        }
    }

    pl_spirv_destroy(&p->spirv);

    // Release streaming ring buffers
    if (p->vbuf.buf)
        p->vbuf.buf->release();
    if (p->ibuf.buf)
        p->ibuf.buf->release();

    if (p->sync_event)
        p->sync_event->release();
    if (p->pending_cb)
        p->pending_cb->release();
    delete p->cb_failed;
    if (p->queue)
        p->queue->release();
    if (p->device)
        p->device->release();

    pl_free((void *) gpu);
}

extern "C"
pl_metal pl_metal_get(pl_gpu gpu)
{
    const struct pl_gpu_fns *impl = (const struct pl_gpu_fns *) PL_PRIV(gpu);
    if (impl->destroy == metal_gpu_destroy) {
        struct pl_gpu_metal *p = (struct pl_gpu_metal *) impl;
        return p->ctx->metal;
    }

    return NULL;
}

static struct pl_gpu_fns pl_fns_metal = {
    .destroy                = metal_gpu_destroy,
    .tex_destroy            = pl_metal_tex_destroy,
    .buf_destroy            = pl_metal_buf_destroy,
    .pass_destroy           = pl_metal_pass_destroy,
    .timer_destroy          = metal_timer_destroy,
    .tex_create             = pl_metal_tex_create,
    .tex_clear_ex           = pl_metal_tex_clear_ex,
    .tex_blit               = pl_metal_tex_blit,
    .tex_upload             = pl_metal_tex_upload,
    .tex_download           = pl_metal_tex_download,
    .tex_poll               = pl_metal_tex_poll,
    .buf_create             = pl_metal_buf_create,
    .buf_write              = pl_metal_buf_write,
    .buf_read               = pl_metal_buf_read,
    .buf_copy               = pl_metal_buf_copy,
    .buf_poll               = pl_metal_buf_poll,
    .desc_namespace         = metal_desc_namespace,
    .pass_create            = pl_metal_pass_create,
    .pass_run               = pl_metal_pass_run,
    .timer_create           = metal_timer_create,
    .timer_query            = metal_timer_query,
    .gpu_flush              = metal_gpu_flush,
    .gpu_finish             = metal_gpu_finish,
    .gpu_is_failed          = metal_gpu_is_failed,
};

extern "C"
pl_gpu pl_gpu_create_metal(struct metal_ctx *ctx)
{
    metal_autorelease pool;
    MTL::Device *device = (MTL::Device *) ctx->metal->device;
    pl_assert(device);

    struct pl_gpu_t *gpu = pl_zalloc_obj(NULL, gpu, struct pl_gpu_metal);
    gpu->log = ctx->log;

    struct pl_gpu_metal *p = (struct pl_gpu_metal *) PL_PRIV(gpu);
    *p = (struct pl_gpu_metal) {
        .impl = pl_fns_metal,
        .ctx = ctx,
        .device = device,
        .queue = (MTL::CommandQueue *) NULL, // set below
    };

    // SPIR-V version (declared before any goto)
    uint32_t spirv_ver = PL_MIN(SPV_VERSION, PL_MAX_SPIRV_VER);
    size_t max_buf = (size_t) device->maxBufferLength();

    // Retain: gpu holds its own reference (released in metal_gpu_destroy).
    // pl_metal_create also retains user-provided devices (released in pl_metal_destroy).
    // User device: +2 retains, -2 releases = balanced.
    // System device: refcount 1 + 1 retain - 2 releases = balanced.
    device->retain();

    // Create command queue
    p->queue = device->newCommandQueue();
    if (!p->queue) {
        PL_FATAL(gpu, "Failed creating Metal command queue");
        goto error;
    }

    // Create initial command buffer
    p->pending_cb = p->queue->commandBuffer();
    if (!p->pending_cb) {
        PL_FATAL(gpu, "Failed creating Metal command buffer");
        goto error;
    }
    p->pending_cb->retain(); // prevent autorelease

    // Per-resource synchronization via SharedEvent generation counter
    p->sync_event = device->newSharedEvent();
    p->cb_failed = new std::shared_ptr<std::atomic<bool>>(
        std::make_shared<std::atomic<bool>>(false));
    if (!p->sync_event) {
        PL_FATAL(gpu, "Failed creating Metal shared event");
        goto error;
    }
    p->current_gen = 1;

    // SPIR-V compiler (needed for GLSL → SPIR-V → MSL pipeline)
    p->spirv = pl_spirv_create(ctx->log, (struct pl_spirv_version) {
        .env_version = pl_spirv_version_to_vulkan(spirv_ver),
        .spv_version = spirv_ver,
    });
    if (!p->spirv) {
        PL_FATAL(gpu, "Failed creating SPIR-V compiler");
        goto error;
    }

    gpu->glsl = (struct pl_glsl_version) {
        .version = 450,
        .vulkan = true,
    };

    gpu->limits = (struct pl_gpu_limits) {
        .max_buf_size       = max_buf,
        .max_ubo_size       = 64 * 1024,
        .max_ssbo_size      = max_buf,
        .max_vbo_size       = max_buf,
        .max_mapped_size    = max_buf,
        .host_cached        = true,
        .max_mapped_vram    = max_buf,
        .max_tex_1d_dim     = 16384,
        .max_tex_2d_dim     = 16384,
        .max_tex_3d_dim     = 2048,
        .buf_transfer       = true,
        .align_tex_xfer_pitch  = 1,
        .align_tex_xfer_offset = 16,
        .max_variable_comps = 0,
        .max_pushc_size     = 4096,
        .align_vertex_stride   = 1,
        .max_dispatch       = { 65535, 65535, 65535 },
        .fragment_queues    = 1,
        .compute_queues     = 1,
    };

    gpu->glsl.compute = true;
    gpu->glsl.max_shmem_size = 32 * 1024; // 32 KiB
    gpu->glsl.max_group_threads = 1024;
    gpu->glsl.max_group_size[0] = 1024;
    gpu->glsl.max_group_size[1] = 1024;
    gpu->glsl.max_group_size[2] = 64;

    pl_metal_setup_formats(gpu);

    gpu->export_caps.tex = PL_HANDLE_MTL_TEX;
    gpu->import_caps.tex = PL_HANDLE_MTL_TEX | PL_HANDLE_IOSURFACE;

    PL_INFO(gpu, "Metal GPU device: %s", device->name()->utf8String());

    return pl_gpu_finalize(gpu);

error:
    metal_gpu_destroy(gpu);
    return NULL;
}
