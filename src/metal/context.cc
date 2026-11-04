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

const struct pl_metal_params pl_metal_default_params = {0};

extern "C"
pl_metal pl_metal_create(pl_log log, const struct pl_metal_params *params)
{
    params = PL_DEF(params, &pl_metal_default_params);
    bool success = false;

    struct pl_metal_t *metal = pl_zalloc_obj(NULL, metal, struct metal_ctx);
    struct metal_ctx *ctx = (struct metal_ctx *) PL_PRIV(metal);
    ctx->log = log;
    ctx->metal = metal;

    MTL::Device *device = (MTL::Device *) params->device;
    if (device) {
        device->retain();
    } else {
        device = MTL::CreateSystemDefaultDevice();
    }

    if (!device) {
        PL_FATAL(ctx, "Failed creating Metal device");
        goto error;
    }

    metal->device = device;

    // Only Apple7+ GPUs (A14, M1 and later) are supported: Intel and AMD GPUs
    // cannot use shared storage for textures, and the advertised limits and
    // format caps assume Apple7. Fail here so the caller can fall back to
    // another API (e.g. Vulkan via MoltenVK).
    if (!device->supportsFamily(MTL::GPUFamilyApple7)) {
        PL_ERR(ctx, "Metal device '%s' is not an Apple7 or newer GPU, which "
               "is required", device->name()->utf8String());
        goto error;
    }

    metal->gpu = pl_gpu_create_metal(ctx);
    if (!metal->gpu)
        goto error;

    success = true;

error:
    if (!success) {
        PL_FATAL(ctx, "Failed initializing Metal device");
        pl_metal_destroy((pl_metal *) &metal);
    }
    return metal;
}

extern "C"
void pl_metal_destroy(pl_metal *pmetal)
{
    pl_metal metal = *pmetal;
    if (!metal)
        return;

    pl_gpu_destroy(metal->gpu);

    if (metal->device)
        ((MTL::Device *) metal->device)->release();

    pl_free_ptr((void **) pmetal);
}
