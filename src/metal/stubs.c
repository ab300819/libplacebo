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

#include "common.h"

const struct pl_metal_params pl_metal_default_params = {0};

pl_metal pl_metal_create(pl_log log, const struct pl_metal_params *params)
{
    (void) params;
    pl_fatal(log, "libplacebo compiled without Metal support!");
    return NULL;
}

void pl_metal_destroy(pl_metal *pmetal)
{
    pl_metal metal = *pmetal;
    pl_assert(!metal);
}

pl_metal pl_metal_get(pl_gpu gpu)
{
    (void) gpu;
    return NULL;
}

pl_swapchain pl_metal_create_swapchain(pl_metal metal,
                                       const struct pl_metal_swapchain_params *params)
{
    (void) metal;
    (void) params;
    pl_unreachable();
}

pl_tex pl_metal_wrap(pl_gpu gpu, const struct pl_metal_wrap_params *params)
{
    (void) gpu;
    (void) params;
    pl_unreachable();
}
