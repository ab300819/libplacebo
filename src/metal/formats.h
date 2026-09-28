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

#include "common.h"

#ifdef __cplusplus
#include <Metal/Metal.hpp>
#endif

struct mtl_format {
    uint32_t mtlfmt;
    uint32_t mtlvtx; // MTL::VertexFormatInvalid if not a vertex format
    struct pl_fmt_t fmt;
};

extern const struct mtl_format pl_metal_formats[];

#ifdef __cplusplus
static inline MTL::PixelFormat fmt_to_mtl(pl_fmt fmt)
{
    const struct mtl_format **fmtp = (const struct mtl_format **) PL_PRIV(fmt);
    return (MTL::PixelFormat) (*fmtp)->mtlfmt;
}

static inline MTL::VertexFormat fmt_to_mtl_vertex(pl_fmt fmt)
{
    const struct mtl_format **fmtp = (const struct mtl_format **) PL_PRIV(fmt);
    return (MTL::VertexFormat) (*fmtp)->mtlvtx;
}
#endif

// Forward-declared setup function (implemented in formats.cc)
#ifdef __cplusplus
extern "C"
#endif
void pl_metal_setup_formats(struct pl_gpu_t *gpu);
