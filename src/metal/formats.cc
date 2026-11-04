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

#define IDX(...) {__VA_ARGS__}
#define BITS(...) {__VA_ARGS__}

#define FMT(_name, _mtl, _vtx, _type, num, size, bits, order) \
    (struct mtl_format) {                                \
        .mtlfmt = (uint32_t) MTL::PixelFormat##_mtl,     \
        .mtlvtx = (uint32_t) MTL::VertexFormat##_vtx,    \
        .fmt = {                                          \
            .name = _name,                                \
            .type = PL_FMT_##_type,                       \
            .num_components  = num,                       \
            .component_depth = bits,                      \
            .internal_size   = size,                      \
            .texel_size      = size,                      \
            .texel_align     = size, /* Metal pitches */  \
            .host_bits       = bits,                      \
            .sample_order    = order,                     \
        },                                                \
    }

#define REGFMT(name, mtl, vtx, type, num, bits)         \
    FMT(name, mtl, vtx, type, num, (num) * (bits) / 8,  \
        BITS(bits, bits, bits, bits),                    \
        IDX(0, 1, 2, 3))

const struct mtl_format pl_metal_formats[] = {
    REGFMT("r8",       R8Unorm,          UCharNormalized,        UNORM, 1,  8),
    REGFMT("rg8",      RG8Unorm,         UChar2Normalized,       UNORM, 2,  8),
    REGFMT("rgba8",    RGBA8Unorm,       UChar4Normalized,       UNORM, 4,  8),
    FMT("bgra8",       BGRA8Unorm,       UChar4Normalized_BGRA,  UNORM, 4, 4,
        BITS(8, 8, 8, 8), IDX(2, 1, 0, 3)),
    REGFMT("r16",      R16Unorm,         UShortNormalized,       UNORM, 1, 16),
    REGFMT("rg16",     RG16Unorm,        UShort2Normalized,      UNORM, 2, 16),
    REGFMT("rgba16",   RGBA16Unorm,      UShort4Normalized,      UNORM, 4, 16),

    REGFMT("r8s",      R8Snorm,          CharNormalized,         SNORM, 1,  8),
    REGFMT("rg8s",     RG8Snorm,         Char2Normalized,        SNORM, 2,  8),
    REGFMT("rgba8s",   RGBA8Snorm,       Char4Normalized,        SNORM, 4,  8),
    REGFMT("r16s",     R16Snorm,         ShortNormalized,        SNORM, 1, 16),
    REGFMT("rg16s",    RG16Snorm,        Short2Normalized,       SNORM, 2, 16),
    REGFMT("rgba16s",  RGBA16Snorm,      Short4Normalized,       SNORM, 4, 16),

    REGFMT("r16hf",    R16Float,         Invalid,                FLOAT, 1, 16),
    REGFMT("rg16hf",   RG16Float,        Invalid,                FLOAT, 2, 16),
    REGFMT("rgba16hf", RGBA16Float,      Invalid,                FLOAT, 4, 16),
    REGFMT("r32f",     R32Float,         Float,                  FLOAT, 1, 32),
    REGFMT("rg32f",    RG32Float,        Float2,                 FLOAT, 2, 32),
    REGFMT("rgb32f",   Invalid,          Float3,                 FLOAT, 3, 32),
    REGFMT("rgba32f",  RGBA32Float,      Float4,                 FLOAT, 4, 32),

    REGFMT("r8u",      R8Uint,           Invalid,                UINT,  1,  8),
    REGFMT("rg8u",     RG8Uint,          Invalid,                UINT,  2,  8),
    REGFMT("rgba8u",   RGBA8Uint,        Invalid,                UINT,  4,  8),
    REGFMT("r16u",     R16Uint,          UShort,                 UINT,  1, 16),
    REGFMT("rg16u",    RG16Uint,         UShort2,                UINT,  2, 16),
    REGFMT("rgba16u",  RGBA16Uint,       UShort4,                UINT,  4, 16),
    REGFMT("r32u",     R32Uint,          UInt,                   UINT,  1, 32),
    REGFMT("rg32u",    RG32Uint,         UInt2,                  UINT,  2, 32),
    REGFMT("rgb32u",   Invalid,          UInt3,                  UINT,  3, 32),
    REGFMT("rgba32u",  RGBA32Uint,       UInt4,                  UINT,  4, 32),

    REGFMT("r8i",      R8Sint,           Invalid,                SINT,  1,  8),
    REGFMT("rg8i",     RG8Sint,          Invalid,                SINT,  2,  8),
    REGFMT("rgba8i",   RGBA8Sint,        Invalid,                SINT,  4,  8),
    REGFMT("r16i",     R16Sint,          Short,                  SINT,  1, 16),
    REGFMT("rg16i",    RG16Sint,         Short2,                 SINT,  2, 16),
    REGFMT("rgba16i",  RGBA16Sint,       Short4,                 SINT,  4, 16),
    REGFMT("r32i",     R32Sint,          Int,                    SINT,  1, 32),
    REGFMT("rg32i",    RG32Sint,         Int2,                   SINT,  2, 32),
    REGFMT("rgb32i",   Invalid,          Int3,                   SINT,  3, 32),
    REGFMT("rgba32i",  RGBA32Sint,       Int4,                   SINT,  4, 32),

    FMT("rgb10a2",     RGB10A2Unorm,     Invalid,                UNORM, 4, 4,
        BITS(10, 10, 10, 2), IDX(0, 1, 2, 3)),
    FMT("rgb10a2u",    RGB10A2Uint,      Invalid,                UINT,  4, 4,
        BITS(10, 10, 10, 2), IDX(0, 1, 2, 3)),
    FMT("rg11b10f",    RG11B10Float,     Invalid,                FLOAT, 3, 4,
        BITS(11, 11, 10), IDX(0, 1, 2)),

    {}
};

#undef EMUFMT
#undef REGFMT
#undef FMT
#undef BITS
#undef IDX

// Formats Metal guarantees `access::read_write` for, per readWriteTextureSupport
static bool fmt_is_readwrite(MTL::PixelFormat fmt, MTL::ReadWriteTextureTier tier)
{
    switch (fmt) {
    case MTL::PixelFormatR32Float:
    case MTL::PixelFormatR32Uint:
    case MTL::PixelFormatR32Sint:
        return tier >= MTL::ReadWriteTextureTier1;
    case MTL::PixelFormatRGBA32Float:
    case MTL::PixelFormatRGBA32Uint:
    case MTL::PixelFormatRGBA32Sint:
    case MTL::PixelFormatRGBA16Float:
    case MTL::PixelFormatRGBA16Uint:
    case MTL::PixelFormatRGBA16Sint:
    case MTL::PixelFormatRGBA8Unorm:
    case MTL::PixelFormatRGBA8Uint:
    case MTL::PixelFormatRGBA8Sint:
    case MTL::PixelFormatR16Float:
    case MTL::PixelFormatR16Uint:
    case MTL::PixelFormatR16Sint:
    case MTL::PixelFormatR8Unorm:
    case MTL::PixelFormatR8Uint:
    case MTL::PixelFormatR8Sint:
        return tier >= MTL::ReadWriteTextureTier2;
    default:
        return false;
    }
}

extern "C"
void pl_metal_setup_formats(struct pl_gpu_t *gpu)
{
    struct pl_gpu_metal *p = (struct pl_gpu_metal *) PL_PRIV(gpu);
    MTL::ReadWriteTextureTier rw_tier = p->device->readWriteTextureSupport();
    bool float32_filter = p->device->supports32BitFloatFiltering();
    // One slot per table entry is enough; skipped formats leave theirs unused
    gpu->formats = pl_calloc_ptr(gpu, PL_ARRAY_SIZE(pl_metal_formats),
                                 gpu->formats);
    gpu->num_formats = 0;

    bool has_compute = gpu->glsl.compute;

    for (int i = 0; pl_metal_formats[i].fmt.name; i++) {
        const struct mtl_format *mtl_fmt = &pl_metal_formats[i];
        bool vertex_only = (mtl_fmt->mtlfmt == 0); // no Metal pixel format

        struct pl_fmt_t *fmt = pl_alloc_obj(gpu, fmt, struct mtl_format *);
        const struct mtl_format **fmtp = (const struct mtl_format **) PL_PRIV(fmt);
        *fmt = mtl_fmt->fmt;
        *fmtp = mtl_fmt;

        fmt->signature = mtl_fmt->mtlfmt;

        for (int j = fmt->num_components; j < 4; j++) {
            fmt->component_depth[j] = 0;
            fmt->sample_order[j] = 0;
            fmt->host_bits[j] = 0;
        }

        if (vertex_only) {
            // Vertex-only formats have no texture capabilities
            fmt->caps = (enum pl_fmt_caps) 0;
        } else {
            fmt->caps = (enum pl_fmt_caps) (fmt->caps | PL_FMT_CAP_SAMPLEABLE);
            bool float32 = fmt->type == PL_FMT_FLOAT &&
                           fmt->component_depth[0] == 32;
            if (fmt->type != PL_FMT_UINT && fmt->type != PL_FMT_SINT &&
                (!float32 || float32_filter))
                fmt->caps = (enum pl_fmt_caps) (fmt->caps | PL_FMT_CAP_LINEAR);

            if (fmt->type == PL_FMT_UNORM || fmt->type == PL_FMT_SNORM ||
                fmt->type == PL_FMT_FLOAT)
            {
                fmt->caps = (enum pl_fmt_caps) (fmt->caps | PL_FMT_CAP_RENDERABLE);
                // Metal supports blending for all renderable formats
                fmt->caps = (enum pl_fmt_caps) (fmt->caps | PL_FMT_CAP_BLENDABLE);
            }

            if (has_compute) {
                fmt->caps = (enum pl_fmt_caps) (fmt->caps | PL_FMT_CAP_STORABLE);
                if (fmt_is_readwrite((MTL::PixelFormat) mtl_fmt->mtlfmt, rw_tier))
                    fmt->caps = (enum pl_fmt_caps) (fmt->caps | PL_FMT_CAP_READWRITE);
            }

            if ((fmt->caps & PL_FMT_CAP_SAMPLEABLE) &&
                (fmt->caps & PL_FMT_CAP_RENDERABLE))
            {
                fmt->caps = (enum pl_fmt_caps) (fmt->caps | PL_FMT_CAP_BLITTABLE);
            }

            fmt->caps = (enum pl_fmt_caps) (fmt->caps | PL_FMT_CAP_HOST_READABLE);
        }

        if (fmt_to_mtl_vertex(fmt) != MTL::VertexFormatInvalid)
            fmt->caps = (enum pl_fmt_caps) (fmt->caps | PL_FMT_CAP_VERTEX);

        if (fmt->caps & (PL_FMT_CAP_VERTEX | PL_FMT_CAP_TEXEL_UNIFORM |
                         PL_FMT_CAP_TEXEL_STORAGE))
        {
            fmt->glsl_type = pl_var_glsl_type_name(pl_var_from_fmt(fmt, ""));
            if (!fmt->glsl_type)
                fmt->caps = (enum pl_fmt_caps) (fmt->caps &
                    ~(PL_FMT_CAP_VERTEX | PL_FMT_CAP_TEXEL_UNIFORM |
                      PL_FMT_CAP_TEXEL_STORAGE));
        }

        if (fmt->caps & (PL_FMT_CAP_STORABLE | PL_FMT_CAP_TEXEL_STORAGE)) {
            fmt->glsl_format = pl_fmt_glsl_format(fmt, fmt->num_components);
            if (!fmt->glsl_format)
                fmt->caps = (enum pl_fmt_caps) (fmt->caps &
                    ~(PL_FMT_CAP_STORABLE | PL_FMT_CAP_TEXEL_STORAGE |
                      PL_FMT_CAP_READWRITE));
        }

        if (!(fmt->caps & PL_FMT_CAP_SAMPLEABLE))
            fmt->caps = (enum pl_fmt_caps) (fmt->caps & ~PL_FMT_CAP_LINEAR);

        if (!(fmt->caps & PL_FMT_CAP_STORABLE))
            fmt->caps = (enum pl_fmt_caps) (fmt->caps & ~PL_FMT_CAP_READWRITE);

        if (fmt->caps & PL_FMT_CAP_SAMPLEABLE)
            fmt->gatherable = true;

        fmt->fourcc = pl_fmt_fourcc(fmt);

        if (!fmt->caps) {
            pl_free(fmt);
            continue;
        }

        gpu->formats[gpu->num_formats++] = fmt;
    }
}
