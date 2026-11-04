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

#include "../swapchain.h"

#include <QuartzCore/QuartzCore.hpp>
#include <CoreGraphics/CGColorSpace.h>
#include <objc/message.h>
#include <objc/runtime.h>

struct priv {
    struct pl_sw_fns impl;
    CA::MetalLayer *layer;
    CA::MetalDrawable *drawable;
    pl_tex backbuffer;
    struct pl_color_space current_csp;
    bool auto_fmt; // layer pixel format is ours to pick per colorspace
};

static pl_fmt find_layer_fmt(pl_gpu gpu, MTL::PixelFormat fmt)
{
    for (int i = 0; i < gpu->num_formats; i++) {
        pl_fmt cand = gpu->formats[i];
        if (fmt_to_mtl(cand) == fmt)
            return cand;
    }
    return NULL;
}

static void metal_sw_destroy(pl_swapchain sw)
{
    struct priv *p = (struct priv *) PL_PRIV(sw);
    pl_tex_destroy(sw->gpu, &p->backbuffer);
    if (p->drawable)
        p->drawable->release();
    if (p->layer)
        p->layer->release();
    pl_free((void *) sw);
}

static int metal_sw_latency(pl_swapchain sw)
{
    // nextDrawable() only blocks once every drawable is in flight
    struct priv *p = (struct priv *) PL_PRIV(sw);
    return (int) p->layer->maximumDrawableCount();
}

static bool metal_sw_resize(pl_swapchain sw, int *width, int *height)
{
    struct priv *p = (struct priv *) PL_PRIV(sw);
    if (p->backbuffer) {
        PL_ERR(sw, "Cannot resize Metal swapchain while frame is in progress");
        return false;
    }

    // Per the pl_swapchain contract, a non-zero dimension is a request to
    // resize and 0 means "leave this one alone"; either way report back what
    // the swapchain ended up with.
    //
    // The requested size is in pixels, not points. Callers driving this from a
    // window system must convert accordingly (e.g. SDL_Metal_GetDrawableSize
    // rather than the window size), otherwise the drawable ends up too small
    // on a HiDPI display.
    CGSize size = p->layer->drawableSize();
    if (*width || *height) {
        if (*width)
            size.width = *width;
        if (*height)
            size.height = *height;
        p->layer->setDrawableSize(size);
        size = p->layer->drawableSize();
    }

    *width = (int) size.width;
    *height = (int) size.height;
    return true;
}

// Picks the layer colorspace for a hint, and rewrites the hint's primaries and
// transfer into what that layer colorspace actually is
static CGColorSpaceRef csp_to_cgcolorspace(struct pl_color_space *csp)
{
    if (csp->primaries == PL_COLOR_PRIM_BT_2020) {
        if (csp->transfer == PL_COLOR_TRC_PQ)
            return CGColorSpaceCreateWithName(kCGColorSpaceITUR_2100_PQ);
        if (csp->transfer == PL_COLOR_TRC_HLG)
            return CGColorSpaceCreateWithName(kCGColorSpaceITUR_2100_HLG);
    }

    if (csp->primaries == PL_COLOR_PRIM_DISPLAY_P3) {
        if (csp->transfer == PL_COLOR_TRC_PQ)
            return CGColorSpaceCreateWithName(kCGColorSpaceDisplayP3_PQ);
        if (csp->transfer == PL_COLOR_TRC_HLG)
            return CGColorSpaceCreateWithName(kCGColorSpaceDisplayP3_HLG);
        csp->transfer = PL_COLOR_TRC_SRGB;
        return CGColorSpaceCreateWithName(kCGColorSpaceDisplayP3);
    }

    csp->primaries = PL_COLOR_PRIM_BT_709;
    if (csp->transfer == PL_COLOR_TRC_LINEAR)
        return CGColorSpaceCreateWithName(kCGColorSpaceLinearSRGB);

    csp->transfer = PL_COLOR_TRC_SRGB;
    return CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
}

static void metal_sw_colorspace_hint(pl_swapchain sw, const struct pl_color_space *csp)
{
    struct priv *p = (struct priv *) PL_PRIV(sw);
    struct pl_color_space applied = *csp;
    CGColorSpaceRef cs = csp_to_cgcolorspace(&applied);
    p->current_csp = applied;

    bool needs_edr = applied.transfer == PL_COLOR_TRC_PQ ||
                     applied.transfer == PL_COLOR_TRC_HLG;

    // 8 bits band visibly with PQ/HLG or linear light; keep a format the
    // caller chose themselves
    if (p->auto_fmt) {
        MTL::PixelFormat pf = MTL::PixelFormatBGRA8Unorm;
        if (needs_edr)
            pf = MTL::PixelFormatRGB10A2Unorm;
        else if (applied.transfer == PL_COLOR_TRC_LINEAR)
            pf = MTL::PixelFormatRGBA16Float;
        p->layer->setPixelFormat(pf);
    }

    // CAMetalLayer.wantsExtendedDynamicRangeContent is macOS 10.11+ but only
    // iOS 16+, and is unavailable on tvOS. Probe for it rather than sending an
    // unrecognized selector, which would abort the process.
    SEL edr_sel = sel_registerName("setWantsExtendedDynamicRangeContent:");
    if (class_respondsToSelector(object_getClass((id) p->layer), edr_sel)) {
        ((void (*)(void *, SEL, bool))objc_msgSend)(
            (void *) p->layer, edr_sel, needs_edr);
    } else if (needs_edr) {
        PL_WARN(sw, "CAMetalLayer does not support extended dynamic range "
                "content on this OS version; HDR output may be clipped");
    }

    if (cs) {
        ((void (*)(void *, SEL, CGColorSpaceRef))objc_msgSend)(
            (void *) p->layer,
            sel_registerName("setColorspace:"),
            cs);
        CGColorSpaceRelease(cs);
    }
}

static bool metal_sw_start_frame(pl_swapchain sw, struct pl_swapchain_frame *out_frame)
{
    struct priv *p = (struct priv *) PL_PRIV(sw);
    struct pl_gpu_metal *gpu_p = (struct pl_gpu_metal *) PL_PRIV(sw->gpu);
    if (metal_is_failed(gpu_p))
        return false;
    if (p->backbuffer) {
        PL_ERR(sw, "Attempted calling `pl_swapchain_start_frame` with frame in progress");
        return false;
    }

    // Scoped per call: a pool held open until submit_frame would pop any
    // pool the caller pushed in between. The drawable is retained below.
    metal_autorelease pool;

    p->drawable = p->layer->nextDrawable();
    if (!p->drawable)
        return false;
    p->drawable->retain();

    MTL::Texture *tex = p->drawable->texture();
    MTL::PixelFormat pf = tex->pixelFormat();
    pl_fmt fmt = find_layer_fmt(sw->gpu, pf);
    if (!fmt) {
        PL_ERR(sw, "Unsupported CAMetalLayer pixel format for swapchain");
        p->drawable->release();
        p->drawable = NULL;
        return false;
    }

    struct pl_metal_wrap_params wrap = {
        .texture = (void *) tex,
        .w = (int) tex->width(),
        .h = (int) tex->height(),
        .format = fmt,
        .debug_tag = PL_DEBUG_TAG,
    };
    p->backbuffer = pl_metal_wrap(sw->gpu, &wrap);
    if (!p->backbuffer) {
        p->drawable->release();
        p->drawable = NULL;
        return false;
    }

    int bits = 0;
    for (int i = 0; i < fmt->num_components; i++)
        bits = PL_MAX(bits, fmt->component_depth[i]);

    *out_frame = (struct pl_swapchain_frame) {
        .fbo = p->backbuffer,
        .flipped = false,
        .color_repr = {
            .sys = PL_COLOR_SYSTEM_RGB,
            .levels = PL_COLOR_LEVELS_FULL,
            .alpha = PL_ALPHA_UNKNOWN,
            .bits = {
                .sample_depth = bits,
                .color_depth = bits,
            },
        },
        .color_space = p->current_csp,
    };

    return true;
}

static bool metal_sw_submit_frame(pl_swapchain sw)
{
    struct priv *p = (struct priv *) PL_PRIV(sw);
    struct pl_gpu_metal *gpu_p = (struct pl_gpu_metal *) PL_PRIV(sw->gpu);
    if (!p->backbuffer || !p->drawable)
        return false;

    if (!gpu_p->pending_cb)
        return false;

    metal_autorelease pool;
    gpu_p->pending_cb->presentDrawable(p->drawable);
    bool ok = metal_advance_cmd_buffer(gpu_p, false);
    if (!ok)
        gpu_p->is_failed = true;

    pl_tex_destroy(sw->gpu, &p->backbuffer);
    p->drawable->release();
    p->drawable = NULL;

    return ok && !metal_is_failed(gpu_p);
}

static void metal_sw_swap_buffers(pl_swapchain sw)
{
    (void) sw;
}

static const struct pl_sw_fns metal_swapchain = {
    .destroy = metal_sw_destroy,
    .latency = metal_sw_latency,
    .resize = metal_sw_resize,
    .colorspace_hint = metal_sw_colorspace_hint,
    .start_frame = metal_sw_start_frame,
    .submit_frame = metal_sw_submit_frame,
    .swap_buffers = metal_sw_swap_buffers,
};

extern "C"
pl_swapchain pl_metal_create_swapchain(pl_metal metal,
                                       const struct pl_metal_swapchain_params *params)
{
    static const struct pl_metal_swapchain_params default_params = {0};
    params = PL_DEF(params, &default_params);
    if (!metal || !metal->gpu || !params->layer)
        return NULL;

    struct pl_swapchain_t *sw = pl_zalloc_obj(NULL, sw, struct priv);
    struct priv *p = (struct priv *) PL_PRIV(sw);
    *sw = (struct pl_swapchain_t) {
        .log = metal->gpu->log,
        .gpu = metal->gpu,
    };
    p->impl = metal_swapchain;
    p->layer = (CA::MetalLayer *) params->layer;
    p->layer->retain();
    p->layer->setDevice((MTL::Device *) metal->device);
    p->layer->setFramebufferOnly(false);
    // BGRA8Unorm is CAMetalLayer's default, so treat it as "not chosen"
    MTL::PixelFormat lfmt = p->layer->pixelFormat();
    p->auto_fmt = lfmt == MTL::PixelFormatInvalid ||
                  lfmt == MTL::PixelFormatBGRA8Unorm;
    if (lfmt == MTL::PixelFormatInvalid)
        p->layer->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    p->current_csp = pl_color_space_srgb;

    return sw;
}
