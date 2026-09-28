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

#ifndef LIBPLACEBO_METAL_H_
#define LIBPLACEBO_METAL_H_

#include <libplacebo/gpu.h>
#include <libplacebo/swapchain.h>

// Only Apple GPUs are supported: pl_metal_create() fails on Intel and AMD
// GPUs, in which case callers should fall back to another API such as Vulkan.
// The backend requires a macOS 12.0 or iOS 15.0 deployment target; building
// against an older target disables it at configure time.
//
// HDR output additionally needs CAMetalLayer's extended dynamic range support
// (macOS 10.11, iOS 16.0); where it is missing, the colorspace hint degrades to
// a warning instead of failing.

PL_API_BEGIN

// Structure representing the actual Metal device and associated GPU instance.
// All Metal API pointers are typed as `void *` to avoid requiring Metal
// headers in the public C API. Cast to `id<MTLDevice>` (ObjC) or
// `MTL::Device *` (metal-cpp) as needed.
typedef const struct pl_metal_t {
    pl_gpu gpu;

    // The MTL::Device in use. The user is free to use this for their own
    // purposes, including retaining the device and using it beyond the
    // lifetime of the pl_metal that created it.
    void *device; // MTL::Device * / id<MTLDevice>
} *pl_metal;

struct pl_metal_params {
    // The Metal device to use. Optional. If NULL, libplacebo will create
    // its own device using MTL::CreateSystemDefaultDevice().
    // If set, pl_metal_create will retain the device.
    void *device; // MTL::Device * / id<MTLDevice>
};

#define pl_metal_params(...) (&(struct pl_metal_params) { __VA_ARGS__ })
PL_API extern const struct pl_metal_params pl_metal_default_params;

struct pl_metal_swapchain_params {
    // The CAMetalLayer to use for presentation.
    void *layer; // CA::MetalLayer * / CAMetalLayer *
};

#define pl_metal_swapchain_params(...) (&(struct pl_metal_swapchain_params) { __VA_ARGS__ })

struct pl_metal_wrap_params {
    // The Metal texture to wrap. Must have been created by the same device
    // used by `gpu`. pl_metal_wrap will retain this texture.
    void *texture; // MTL::Texture * / id<MTLTexture>
    int w;
    int h;
    pl_fmt format;
    pl_debug_tag debug_tag;
};

#define pl_metal_wrap_params(...) (&(struct pl_metal_wrap_params) { \
        .debug_tag = PL_DEBUG_TAG,                                     \
        __VA_ARGS__                                                    \
    })

// Creates a new Metal GPU instance. If params is left as NULL, it defaults
// to &pl_metal_default_params. If an existing device is provided in
// params->device, pl_metal_create will retain it.
PL_API pl_metal pl_metal_create(pl_log log, const struct pl_metal_params *params);

// Release the Metal device and all associated resources. All libplacebo
// objects allocated from this pl_metal (via metal->gpu) must be explicitly
// destroyed by the user before calling this.
PL_API void pl_metal_destroy(pl_metal *metal);

// For a pl_gpu backed by pl_metal, retrieve the underlying pl_metal.
// Returns NULL for any other type of gpu.
PL_API pl_metal pl_metal_get(pl_gpu gpu);

// Creates a Metal swapchain wrapping a CAMetalLayer.
PL_API pl_swapchain pl_metal_create_swapchain(pl_metal metal,
                                              const struct pl_metal_swapchain_params *params);

// Wraps an external Metal texture into a pl_tex abstraction.
// May fail due to incompatible formats or other reasons, returning NULL.
PL_API pl_tex pl_metal_wrap(pl_gpu gpu, const struct pl_metal_wrap_params *params);

PL_API_END

#endif // LIBPLACEBO_METAL_H_
