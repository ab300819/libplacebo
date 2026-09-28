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

#pragma once

#include <math.h>
#include <stdint.h>
#include <string.h>

// Mean Squared Error over float buffers
static inline double img_cmp_mse(const float *a, const float *b, int count)
{
    double sum = 0;
    for (int i = 0; i < count; i++) {
        double d = (double) a[i] - (double) b[i];
        sum += d * d;
    }
    return sum / count;
}

// PSNR in dB (peak = 1.0 for normalized float)
// Returns INFINITY if images are identical
static inline double img_cmp_psnr(const float *a, const float *b, int count)
{
    double mse = img_cmp_mse(a, b, count);
    if (mse < 1e-20)
        return INFINITY;
    return 10.0 * log10(1.0 / mse);
}

// Max absolute difference. Blind to NaN by construction -- see
// img_cmp_count_nonfinite().
static inline float img_cmp_max_diff(const float *a, const float *b, int count)
{
    float max = 0;
    for (int i = 0; i < count; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > max)
            max = d;
    }
    return max;
}

// Number of positions where either side is not finite.
//
// Every comparison below is blind to NaN: `NaN > x` is false, so a NaN slips
// past img_cmp_max_diff() (which keeps returning 0) and past the tolerance
// check in img_cmp_count_above(). Callers must therefore establish that this
// returns 0 before trusting any of the other metrics.
static inline int img_cmp_count_nonfinite(const float *a, const float *b,
                                          int count)
{
    int n = 0;
    for (int i = 0; i < count; i++) {
        if (!isfinite(a[i]) || !isfinite(b[i]))
            n++;
    }
    return n;
}

// Number of positions whose IEEE-754 bit patterns differ.
//
// A tolerance of 0 is not the same test: +0.0 and -0.0 are numerically equal
// but have different bit patterns, so only this function can back a claim of
// "bit-identical". Reject non-finite values first -- this would otherwise
// report NaN payload differences, which are rarely what the caller means.
static inline int img_cmp_count_bitdiff(const float *a, const float *b,
                                       int count)
{
    int n = 0;
    for (int i = 0; i < count; i++) {
        uint32_t ua, ub;
        memcpy(&ua, &a[i], sizeof(ua));
        memcpy(&ub, &b[i], sizeof(ub));
        if (ua != ub)
            n++;
    }
    return n;
}

// Number of values differing by more than `tol`, counting non-finite values as
// diverging. MSE and PSNR average over the whole buffer and therefore hide
// localized damage (a few broken rows from a stride bug barely move them), so
// comparisons that care about correctness rather than perceptual similarity
// should bound this instead.
//
// Note that a tolerance of 0 means "numerically equal", not "bit-identical":
// +0.0 and -0.0 compare equal here.
static inline int img_cmp_count_above(const float *a, const float *b, int count,
                                      float tol)
{
    int n = 0;
    for (int i = 0; i < count; i++) {
        if (!isfinite(a[i]) || !isfinite(b[i]) || fabsf(a[i] - b[i]) > tol)
            n++;
    }
    return n;
}
