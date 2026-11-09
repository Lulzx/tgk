#include "tgk.h"
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Check if integer n factors completely into 2, 3, and 5 */
static int is_smooth_235(int n) {
    if (n <= 0) return 0;
    while (n % 2 == 0) n /= 2;
    while (n % 3 == 0) n /= 3;
    while (n % 5 == 0) n /= 5;
    return (n == 1);
}

static int next_smooth_235(int min_n, int must_be_even) {
    int n = min_n;
    if (must_be_even && (n & 1)) n++;
    while (!is_smooth_235(n)) {
        n += (must_be_even ? 2 : 1);
    }
    return n;
}

/* Precompute twiddle table W_N^k = exp(sign * 2*pi*i * k / N) for k = 0..N-1 */
static tgk_complex *make_twiddles(int n, int inv) {
    tgk_complex *tw = tgk_malloc_aligned(sizeof(tgk_complex) * n);
    double sign = inv ? 1.0 : -1.0;
    for (int k = 0; k < n; k++) {
        double theta = sign * (2.0 * M_PI * (double)k) / (double)n;
        tw[k].re = (tgk_real)cos(theta);
        tw[k].im = (tgk_real)sin(theta);
    }
    return tw;
}

int tgk_fft_init(tgk_fft_plan *plan, int nkx, int nky) {
    memset(plan, 0, sizeof(*plan));
    plan->nkx = nkx;
    plan->nky = nky;

    /* 2/3-rule dealiasing physical dimensions:
     * nx_pad >= (3 * nkx) / 2, ny_pad >= 3 * nky, both smooth in {2, 3, 5} */
    int min_nx = (3 * nkx + 1) / 2;
    if (min_nx < 4) min_nx = 4;
    int min_ny = 3 * nky;
    if (min_ny < 4) min_ny = 4;

    plan->nx_pad = next_smooth_235(min_nx, 1);
    plan->ny_pad = next_smooth_235(min_ny, 1);
    plan->ny_r2c = plan->ny_pad / 2 + 1;
    plan->norm   = 1.0 / ((double)plan->nx_pad * (double)plan->ny_pad);

    plan->tw_x_fwd = make_twiddles(plan->nx_pad, 0);
    plan->tw_x_inv = make_twiddles(plan->nx_pad, 1);
    plan->tw_y_fwd = make_twiddles(plan->ny_pad, 0);
    plan->tw_y_inv = make_twiddles(plan->ny_pad, 1);
    plan->tw_r2c   = make_twiddles(plan->ny_pad, 0);

    return 0;
}

void tgk_fft_free(tgk_fft_plan *plan) {
    tgk_free_aligned(plan->tw_x_fwd);
    tgk_free_aligned(plan->tw_x_inv);
    tgk_free_aligned(plan->tw_y_fwd);
    tgk_free_aligned(plan->tw_y_inv);
    tgk_free_aligned(plan->tw_r2c);
    memset(plan, 0, sizeof(*plan));
}

/* Vectorized Stockham auto-sorting mixed-radix (4, 2, 3, 5) 1D complex-to-complex FFT.
 * Uses radix-4 when divisible by 4, and NEON FCMLA vectorization across j. */
static void fft_1d_stockham(const tgk_complex * restrict in, tgk_complex * restrict out,
                            tgk_complex * restrict tmp, int n, const tgk_complex * restrict tw, int inv) {
    if (n == 1) {
        out[0] = in[0];
        return;
    }

    const tgk_real c3_1 = (tgk_real)(-0.5);
    const tgk_real c3_2 = (tgk_real)(inv ? 0.86602540378443864676 : -0.86602540378443864676);

    const tgk_real c5_1 = (tgk_real)(0.30901699437494742410);
    const tgk_real c5_2 = (tgk_real)(inv ? 0.95105651629515357212 : -0.95105651629515357212);
    const tgk_real c5_3 = (tgk_real)(-0.80901699437494742410);
    const tgk_real c5_4 = (tgk_real)(inv ? 0.58778525229247312917 : -0.58778525229247312917);

    const tgk_real rot_sign = (tgk_real)(inv ? 1.0 : -1.0);

    const tgk_complex *src = in;
    tgk_complex *dst = tmp;

    int m = 1;
    int rem = n;

    while (rem > 1) {
        int r;
        if ((rem & 3) == 0) r = 4;
        else if ((rem & 1) == 0) r = 2;
        else if (rem % 3 == 0) r = 3;
        else r = 5;

        int l = rem / r;
        int stride_tw = l;

        if (r == 4) {
            if (m == 1) {
                for (int k = 0; k < l; k++) {
                    tgk_complex x0 = src[k];
                    tgk_complex x1 = src[k + l];
                    tgk_complex x2 = src[k + 2 * l];
                    tgk_complex x3 = src[k + 3 * l];

                    tgk_complex t0 = tgk_cadd(x0, x2);
                    tgk_complex t1 = tgk_csub(x0, x2);
                    tgk_complex t2 = tgk_cadd(x1, x3);
                    tgk_complex t3 = tgk_csub(x1, x3);
                    /* i * rot_sign * t3 = (-rot_sign * t3.im, rot_sign * t3.re) */
                    tgk_complex it3 = {-rot_sign * t3.im, rot_sign * t3.re};

                    dst[4 * k]     = tgk_cadd(t0, t2);
                    dst[4 * k + 1] = tgk_cadd(t1, it3);
                    dst[4 * k + 2] = tgk_csub(t0, t2);
                    dst[4 * k + 3] = tgk_csub(t1, it3);
                }
            } else {
                for (int k = 0; k < l; k++) {
                    const tgk_complex *s0 = src + k * m;
                    const tgk_complex *s1 = src + (k + l) * m;
                    const tgk_complex *s2 = src + (k + 2 * l) * m;
                    const tgk_complex *s3 = src + (k + 3 * l) * m;
                    tgk_complex *d0 = dst + k * (4 * m);
                    tgk_complex *d1 = d0 + m;
                    tgk_complex *d2 = d0 + 2 * m;
                    tgk_complex *d3 = d0 + 3 * m;

                    int j = 0;
#if TGK_HAS_NEON && !TGK_REAL_IS_FP64
                    for (; j + 2 <= m; j += 2) {
                        tgk_vc2 x0 = tgk_vc2_load(s0 + j);
                        tgk_vc2 x1 = tgk_vc2_load(s1 + j);
                        tgk_vc2 x2 = tgk_vc2_load(s2 + j);
                        tgk_vc2 x3 = tgk_vc2_load(s3 + j);

                        int j0 = j * stride_tw, j1 = (j + 1) * stride_tw;
                        float32x4_t w1v = {tw[j0].re, tw[j0].im, tw[j1].re, tw[j1].im};
                        float32x4_t w2v = {tw[(2 * j0) % n].re, tw[(2 * j0) % n].im,
                                           tw[(2 * j1) % n].re, tw[(2 * j1) % n].im};
                        float32x4_t w3v = {tw[(3 * j0) % n].re, tw[(3 * j0) % n].im,
                                           tw[(3 * j1) % n].re, tw[(3 * j1) % n].im};

                        x1 = tgk_vc2_mul(w1v, x1);
                        x2 = tgk_vc2_mul(w2v, x2);
                        x3 = tgk_vc2_mul(w3v, x3);

                        tgk_vc2 t0 = tgk_vc2_add(x0, x2);
                        tgk_vc2 t1 = tgk_vc2_sub(x0, x2);
                        tgk_vc2 t2 = tgk_vc2_add(x1, x3);
                        tgk_vc2 t3 = tgk_vc2_sub(x1, x3);
                        tgk_vc2 it3 = tgk_vc2_mul_is2(rot_sign, rot_sign, t3);

                        tgk_vc2_store(d0 + j, tgk_vc2_add(t0, t2));
                        tgk_vc2_store(d1 + j, tgk_vc2_add(t1, it3));
                        tgk_vc2_store(d2 + j, tgk_vc2_sub(t0, t2));
                        tgk_vc2_store(d3 + j, tgk_vc2_sub(t1, it3));
                    }
#endif
                    for (; j < m; j++) {
                        int jt = j * stride_tw;
                        tgk_complex x0 = s0[j];
                        tgk_complex x1 = tgk_cmul(tw[jt], s1[j]);
                        tgk_complex x2 = tgk_cmul(tw[(2 * jt) % n], s2[j]);
                        tgk_complex x3 = tgk_cmul(tw[(3 * jt) % n], s3[j]);

                        tgk_complex t0 = tgk_cadd(x0, x2);
                        tgk_complex t1 = tgk_csub(x0, x2);
                        tgk_complex t2 = tgk_cadd(x1, x3);
                        tgk_complex t3 = tgk_csub(x1, x3);
                        tgk_complex it3 = {-rot_sign * t3.im, rot_sign * t3.re};

                        d0[j] = tgk_cadd(t0, t2);
                        d1[j] = tgk_cadd(t1, it3);
                        d2[j] = tgk_csub(t0, t2);
                        d3[j] = tgk_csub(t1, it3);
                    }
                }
            }
        } else if (r == 2) {
            if (m == 1) {
                /* First pass: w = 1, pure add/sub across k = 0 .. l-1 */
                int k = 0;
#if TGK_HAS_NEON && !TGK_REAL_IS_FP64
                for (; k + 2 <= l; k += 2) {
                    tgk_vc2 a = tgk_vc2_load(src + k);
                    tgk_vc2 b = tgk_vc2_load(src + k + l);
                    tgk_vc2 sum = tgk_vc2_add(a, b);
                    tgk_vc2 dif = tgk_vc2_sub(a, b);
                    /* Interleave [sum0, dif0, sum1, dif1] */
                    float32x4x2_t zip = vuzpq_f32(vtrn1q_f32(sum, dif), vtrn2q_f32(sum, dif));
                    (void)zip;
                    dst[2 * k]     = (tgk_complex){vgetq_lane_f32(sum, 0), vgetq_lane_f32(sum, 1)};
                    dst[2 * k + 1] = (tgk_complex){vgetq_lane_f32(dif, 0), vgetq_lane_f32(dif, 1)};
                    dst[2 * k + 2] = (tgk_complex){vgetq_lane_f32(sum, 2), vgetq_lane_f32(sum, 3)};
                    dst[2 * k + 3] = (tgk_complex){vgetq_lane_f32(dif, 2), vgetq_lane_f32(dif, 3)};
                }
#endif
                for (; k < l; k++) {
                    tgk_complex a = src[k];
                    tgk_complex b = src[k + l];
                    dst[2 * k]     = tgk_cadd(a, b);
                    dst[2 * k + 1] = tgk_csub(a, b);
                }
            } else {
                /* Outer loop k, inner loop j (contiguous in memory!) */
                for (int k = 0; k < l; k++) {
                    const tgk_complex *s0 = src + k * m;
                    const tgk_complex *s1 = src + (k + l) * m;
                    tgk_complex *d0 = dst + k * (2 * m);
                    tgk_complex *d1 = d0 + m;
                    int j = 0;
#if TGK_HAS_NEON && !TGK_REAL_IS_FP64
                    for (; j + 2 <= m; j += 2) {
                        tgk_vc2 a = tgk_vc2_load(s0 + j);
                        tgk_vc2 b = tgk_vc2_load(s1 + j);
                        tgk_complex w0 = tw[j * stride_tw];
                        tgk_complex w1 = tw[(j + 1) * stride_tw];
                        float32x4_t wv = {w0.re, w0.im, w1.re, w1.im};
                        tgk_vc2 wb = tgk_vc2_mul(wv, b);
                        tgk_vc2_store(d0 + j, tgk_vc2_add(a, wb));
                        tgk_vc2_store(d1 + j, tgk_vc2_sub(a, wb));
                    }
#endif
                    for (; j < m; j++) {
                        tgk_complex w = tw[j * stride_tw];
                        tgk_complex a = s0[j];
                        tgk_complex wb = tgk_cmul(w, s1[j]);
                        d0[j] = tgk_cadd(a, wb);
                        d1[j] = tgk_csub(a, wb);
                    }
                }
            }
        } else if (r == 3) {
            for (int k = 0; k < l; k++) {
                const tgk_complex *s0 = src + k * m;
                const tgk_complex *s1 = src + (k + l) * m;
                const tgk_complex *s2 = src + (k + 2 * l) * m;
                tgk_complex *d0 = dst + k * (3 * m);
                tgk_complex *d1 = d0 + m;
                tgk_complex *d2 = d0 + 2 * m;
                int j = 0;
#if TGK_HAS_NEON && !TGK_REAL_IS_FP64
                for (; j + 2 <= m; j += 2) {
                    int j0 = j * stride_tw, j1 = (j + 1) * stride_tw;
                    float32x4_t w1v = {tw[j0].re, tw[j0].im, tw[j1].re, tw[j1].im};
                    float32x4_t w2v = {tw[(2 * j0) % n].re, tw[(2 * j0) % n].im,
                                       tw[(2 * j1) % n].re, tw[(2 * j1) % n].im};
                    tgk_vc2 x0 = tgk_vc2_load(s0 + j);
                    tgk_vc2 x1 = tgk_vc2_mul(w1v, tgk_vc2_load(s1 + j));
                    tgk_vc2 x2 = tgk_vc2_mul(w2v, tgk_vc2_load(s2 + j));

                    tgk_vc2 t1 = tgk_vc2_add(x1, x2);
                    tgk_vc2 t2 = tgk_vc2_sub(x1, x2);
                    tgk_vc2 m1 = tgk_vc2_fma_real(x0, c3_1, t1);
                    tgk_vc2 m2 = tgk_vc2_mul_is2(c3_2, c3_2, t2);

                    tgk_vc2_store(d0 + j, tgk_vc2_add(x0, t1));
                    tgk_vc2_store(d1 + j, tgk_vc2_add(m1, m2));
                    tgk_vc2_store(d2 + j, tgk_vc2_sub(m1, m2));
                }
#endif
                for (; j < m; j++) {
                    tgk_complex w1 = tw[j * stride_tw];
                    tgk_complex w2 = tw[(2 * j * stride_tw) % n];
                    tgk_complex x0 = s0[j];
                    tgk_complex x1 = tgk_cmul(w1, s1[j]);
                    tgk_complex x2 = tgk_cmul(w2, s2[j]);

                    tgk_complex t1 = tgk_cadd(x1, x2);
                    tgk_complex t2 = tgk_csub(x1, x2);
                    tgk_complex m1 = {x0.re + c3_1 * t1.re, x0.im + c3_1 * t1.im};
                    tgk_complex m2 = {-c3_2 * t2.im, c3_2 * t2.re};

                    d0[j] = tgk_cadd(x0, t1);
                    d1[j] = tgk_cadd(m1, m2);
                    d2[j] = tgk_csub(m1, m2);
                }
            }
        } else { /* r == 5 */
            for (int k = 0; k < l; k++) {
                const tgk_complex *s0 = src + k * m;
                const tgk_complex *s1 = src + (k + 1 * l) * m;
                const tgk_complex *s2 = src + (k + 2 * l) * m;
                const tgk_complex *s3 = src + (k + 3 * l) * m;
                const tgk_complex *s4 = src + (k + 4 * l) * m;
                tgk_complex *d0 = dst + k * (5 * m);
                for (int j = 0; j < m; j++) {
                    tgk_complex w1 = tw[(1 * j * stride_tw) % n];
                    tgk_complex w2 = tw[(2 * j * stride_tw) % n];
                    tgk_complex w3 = tw[(3 * j * stride_tw) % n];
                    tgk_complex w4 = tw[(4 * j * stride_tw) % n];
                    tgk_complex x0 = s0[j];
                    tgk_complex x1 = tgk_cmul(w1, s1[j]);
                    tgk_complex x2 = tgk_cmul(w2, s2[j]);
                    tgk_complex x3 = tgk_cmul(w3, s3[j]);
                    tgk_complex x4 = tgk_cmul(w4, s4[j]);

                    tgk_complex t1 = tgk_cadd(x1, x4);
                    tgk_complex t2 = tgk_cadd(x2, x3);
                    tgk_complex t3 = tgk_csub(x1, x4);
                    tgk_complex t4 = tgk_csub(x2, x3);

                    tgk_complex m0 = tgk_cadd(x0, tgk_cadd(t1, t2));
                    tgk_complex m1 = {x0.re + c5_1 * t1.re + c5_3 * t2.re,
                                      x0.im + c5_1 * t1.im + c5_3 * t2.im};
                    tgk_complex m2 = {x0.re + c5_3 * t1.re + c5_1 * t2.re,
                                      x0.im + c5_3 * t1.im + c5_1 * t2.im};

                    tgk_complex u1 = {-(c5_2 * t3.im + c5_4 * t4.im),
                                       (c5_2 * t3.re + c5_4 * t4.re)};
                    tgk_complex u2 = {-(c5_4 * t3.im - c5_2 * t4.im),
                                       (c5_4 * t3.re - c5_2 * t4.re)};

                    d0[j]         = m0;
                    d0[j + 1 * m] = tgk_cadd(m1, u1);
                    d0[j + 2 * m] = tgk_cadd(m2, u2);
                    d0[j + 3 * m] = tgk_csub(m2, u2);
                    d0[j + 4 * m] = tgk_csub(m1, u1);
                }
            }
        }

        m *= r;
        rem = l;
        if (src == in) {
            src = tmp;
            dst = out;
        } else {
            const tgk_complex *swap = src;
            src = dst;
            dst = (tgk_complex *)swap;
        }
    }

    if (src != out) {
        memcpy(out, src, sizeof(tgk_complex) * n);
    }
}

void tgk_fft_1d_c2c(tgk_complex *in, tgk_complex *out, int n, int inv) {
    tgk_complex *tw  = make_twiddles(n, inv);
    tgk_complex *tmp = tgk_malloc_aligned(sizeof(tgk_complex) * n);
    fft_1d_stockham(in, out, tmp, n, tw, inv);
    tgk_free_aligned(tmp);
    tgk_free_aligned(tw);
}

/* 2D Complex-to-Real synthesis with folded 2/3-rule zero-padding:
 * Input:  spec_in[nky][nkx] (dealiased Fourier coefficients, ky >= 0)
 * Output: real_out[nx_pad][ny_pad] (physical space values on padded grid)
 *
 * Convention:
 * f(x_m, y_l) = sum_{kx, ky} F(kx, ky) * exp(i * (kx * x_m + ky * y_l))
 * Using reality condition F(-kx, -ky) = conj(F(kx, ky)). */
/* Fused 2D complex-to-real gradient synthesis (x-first with zero-skip & Hermitian pair):
 * Given a single spectral field F(kx, ky), computes both real_df_dx = dF/dx(x, y) and
 * real_df_dy = dF/dy(x, y) in physical space on the padded (dealiased) grid.
 *
 * Key insight:
 *   - Step 1: Only (2 * nky - 1) 1D x-FFTs of length nx_pad (for ky=0..nky-1 of i*kx*F,
 *     and ky=1..nky-1 of F scaled by i*ky afterwards).
 *   - Step 2: Pack dF/dx(x, ky) + i * dF/dy(x, ky) into ONE 1D y-FFT of length ny_pad per ix.
 * Total: 63 x-FFTs + 192 y-FFTs! */
void tgk_fft_c2r_grad_dealiased(const tgk_fft_plan *plan, const tgk_complex *spec_f,
                                const double *kx, const double *ky,
                                tgk_real *real_df_dx, tgk_real *real_df_dy,
                                tgk_complex *scratch1, tgk_complex *scratch2) {
    const int nkx    = plan->nkx;
    const int nky    = plan->nky;
    const int nx_pad = plan->nx_pad;
    const int ny_pad = plan->ny_pad;
    const int half_x = nkx / 2;
    const int n_pos  = (nkx > 1) ? (half_x + 1) : 1;

    tgk_complex *row_in  = scratch2;
    tgk_complex *row_out = scratch2 + nx_pad + ny_pad;
    tgk_complex *row_tmp = row_out  + nx_pad + ny_pad;
    tgk_complex *scr_dy  = scratch1 + (size_t)nx_pad * nky;

    for (int iky = 0; iky < nky; iky++) {
        const tgk_complex *sf = spec_f + (size_t)iky * nkx;

        memset(row_in, 0, sizeof(tgk_complex) * nx_pad);
        for (int ikx = 0; ikx < n_pos; ikx++) {
            row_in[ikx] = tgk_cmul_is((tgk_real)kx[ikx], sf[ikx]);
        }
        for (int ikx = half_x + 1; ikx < nkx; ikx++) {
            row_in[nx_pad - nkx + ikx] = tgk_cmul_is((tgk_real)kx[ikx], sf[ikx]);
        }
        fft_1d_stockham(row_in, row_out, row_tmp, nx_pad, plan->tw_x_inv, 1);
        for (int ix = 0; ix < nx_pad; ix++) {
            scratch1[(size_t)ix * nky + iky] = row_out[ix];
        }

        if (iky == 0) {
            for (int ix = 0; ix < nx_pad; ix++) {
                scr_dy[(size_t)ix * nky] = (tgk_complex){0.0f, 0.0f};
            }
        } else {
            tgk_real ky_val = (tgk_real)ky[iky];
            memset(row_in, 0, sizeof(tgk_complex) * nx_pad);
            for (int ikx = 0; ikx < n_pos; ikx++) {
                row_in[ikx] = sf[ikx];
            }
            for (int ikx = half_x + 1; ikx < nkx; ikx++) {
                row_in[nx_pad - nkx + ikx] = sf[ikx];
            }
            fft_1d_stockham(row_in, row_out, row_tmp, nx_pad, plan->tw_x_inv, 1);
            for (int ix = 0; ix < nx_pad; ix++) {
                scr_dy[(size_t)ix * nky + iky] = tgk_cmul_is(ky_val, row_out[ix]);
            }
        }
    }

    for (int ix = 0; ix < nx_pad; ix++) {
        const tgk_complex *ca = scratch1 + (size_t)ix * nky;
        const tgk_complex *cb = scr_dy   + (size_t)ix * nky;

        memset(row_in, 0, sizeof(tgk_complex) * ny_pad);
        row_in[0].re = ca[0].re;
        row_in[0].im = ca[0].im;

        for (int iky = 1; iky < nky; iky++) {
            tgk_complex f0 = ca[iky];
            tgk_complex f1 = cb[iky];
            row_in[iky].re          =  f0.re - f1.im;
            row_in[iky].im          =  f0.im + f1.re;
            row_in[ny_pad - iky].re =  f0.re + f1.im;
            row_in[ny_pad - iky].im = -f0.im + f1.re;
        }

        fft_1d_stockham(row_in, row_out, row_tmp, ny_pad, plan->tw_y_inv, 1);

        tgk_real *da = real_df_dx + (size_t)ix * ny_pad;
        tgk_real *db = real_df_dy + (size_t)ix * ny_pad;
        for (int iy = 0; iy < ny_pad; iy++) {
            da[iy] = row_out[iy].re;
            db[iy] = row_out[iy].im;
        }
    }
}

/* Paired 2D Real-to-Complex analysis for two real fields real_a(x,y) and real_b(x,y) simultaneously
 * (e.g. Poisson bracket at z=iz and z=iz+1), cutting 1D y-FFTs in half! */
void tgk_fft_r2c_pair_dealiased(const tgk_fft_plan *plan, const tgk_real *real_a,
                                const tgk_real *real_b, tgk_complex *spec_a, tgk_complex *spec_b,
                                tgk_complex *scratch1, tgk_complex *scratch2) {
    const int nkx    = plan->nkx;
    const int nky    = plan->nky;
    const int nx_pad = plan->nx_pad;
    const int ny_pad = plan->ny_pad;
    const int half_x = nkx / 2;
    const int n_pos  = (nkx > 1) ? (half_x + 1) : 1;
    const tgk_real scale = (tgk_real)plan->norm;

    tgk_complex *row_in  = scratch2;
    tgk_complex *row_out = scratch2 + nx_pad + ny_pad;
    tgk_complex *row_tmp = row_out  + nx_pad + ny_pad;
    tgk_complex *scr_b   = scratch1 + (size_t)nx_pad * nky;

    /* Step 1: Pack real_a(x, y) + i * real_b(x, y) and run ONE 1D y-FFT per ix */
    for (int ix = 0; ix < nx_pad; ix++) {
        const tgk_real *sa = real_a + (size_t)ix * ny_pad;
        const tgk_real *sb = real_b + (size_t)ix * ny_pad;
        for (int iy = 0; iy < ny_pad; iy++) {
            row_in[iy].re = sa[iy];
            row_in[iy].im = sb[iy];
        }

        fft_1d_stockham(row_in, row_out, row_tmp, ny_pad, plan->tw_y_fwd, 0);

        tgk_complex *ca = scratch1 + (size_t)ix * nky;
        tgk_complex *cb = scr_b    + (size_t)ix * nky;

        ca[0].re = row_out[0].re;
        ca[0].im = 0.0f;
        cb[0].re = row_out[0].im;
        cb[0].im = 0.0f;

        for (int iky = 1; iky < nky; iky++) {
            tgk_complex zp = row_out[iky];
            tgk_complex zm = row_out[ny_pad - iky];
            ca[iky].re = 0.5f * (zp.re + zm.re);
            ca[iky].im = 0.5f * (zp.im - zm.im);
            cb[iky].re = 0.5f * (zp.im + zm.im);
            cb[iky].im = 0.5f * (zm.re - zp.re);
        }
    }

    /* Step 2: 1D forward FFT along x for A and B */
    for (int iky = 0; iky < nky; iky++) {
        for (int ix = 0; ix < nx_pad; ix++) row_in[ix] = scratch1[(size_t)ix * nky + iky];
        fft_1d_stockham(row_in, row_out, row_tmp, nx_pad, plan->tw_x_fwd, 0);
        tgk_complex *da = spec_a + (size_t)iky * nkx;
        for (int ikx = 0; ikx < n_pos; ikx++) da[ikx] = tgk_cscale(scale, row_out[ikx]);
        for (int ikx = half_x + 1; ikx < nkx; ikx++) da[ikx] = tgk_cscale(scale, row_out[nx_pad - nkx + ikx]);
        if (nkx > 1 && (nkx & 1) == 0) da[half_x] = (tgk_complex){0.0f, 0.0f};

        for (int ix = 0; ix < nx_pad; ix++) row_in[ix] = scr_b[(size_t)ix * nky + iky];
        fft_1d_stockham(row_in, row_out, row_tmp, nx_pad, plan->tw_x_fwd, 0);
        tgk_complex *db = spec_b + (size_t)iky * nkx;
        for (int ikx = 0; ikx < n_pos; ikx++) db[ikx] = tgk_cscale(scale, row_out[ikx]);
        for (int ikx = half_x + 1; ikx < nkx; ikx++) db[ikx] = tgk_cscale(scale, row_out[nx_pad - nkx + ikx]);
        if (nkx > 1 && (nkx & 1) == 0) db[half_x] = (tgk_complex){0.0f, 0.0f};
    }

    spec_a[0] = (tgk_complex){0.0f, 0.0f};
    spec_b[0] = (tgk_complex){0.0f, 0.0f};
    for (int ikx = 1; ikx < half_x; ikx++) {
        int ikx_neg = nkx - ikx;
        tgk_complex pa = spec_a[ikx], na = spec_a[ikx_neg];
        spec_a[ikx]     = (tgk_complex){0.5f * (pa.re + na.re),  0.5f * (pa.im - na.im)};
        spec_a[ikx_neg] = (tgk_complex){0.5f * (pa.re + na.re), -0.5f * (pa.im - na.im)};

        tgk_complex pb = spec_b[ikx], nb = spec_b[ikx_neg];
        spec_b[ikx]     = (tgk_complex){0.5f * (pb.re + nb.re),  0.5f * (pb.im - nb.im)};
        spec_b[ikx_neg] = (tgk_complex){0.5f * (pb.re + nb.re), -0.5f * (pb.im - nb.im)};
    }
}

void tgk_fft_c2r_pair_dealiased(const tgk_fft_plan *plan, const tgk_complex *spec_a,
                                const tgk_complex *spec_b, tgk_real *real_a, tgk_real *real_b,
                                tgk_complex *scratch1, tgk_complex *scratch2) {
    const int nkx    = plan->nkx;
    const int nky    = plan->nky;
    const int nx_pad = plan->nx_pad;
    const int ny_pad = plan->ny_pad;
    const int half_x = nkx / 2;

    tgk_complex *row_in  = scratch2;
    tgk_complex *row_out = scratch2 + nx_pad + ny_pad;
    tgk_complex *row_tmp = row_out  + nx_pad + ny_pad;
    tgk_complex *scr_b   = scratch1 + (size_t)nx_pad * nky;

    /* Step 1: For each pair of ky modes (iky, iky+1), run TWO 1D inverse FFTs in x simultaneously
     * by packing them or running back-to-back with shared twiddles, and derive BOTH
     * dF/dx(x, ky) = FFT_x^{-1}[i*kx*F] and dF/dy(x, ky) = i*ky * FFT_x^{-1}[F]!
     * Wait: notice that for dF/dy, the factor i*ky does NOT depend on kx at all!
     * That means if we know F(kx, ky), then dF/dy(x, ky) = i * ky * F(x, ky),
     * AND if we instead transform F(kx, ky) in x first (just ONE 1D x-FFT of length nx_pad per ky!),
     * or if we transform F(kx, ky) and i*kx*F(kx, ky), can we get dF/dx from F(x, ky)?
     * Even better: in `tgk_fft_c2r_grad_dealiased`, the input is a single complex spectrum F(kx, ky),
     * and we want BOTH real_dx = dF/dx(x,y) and real_dy = dF/dy(x,y)!
     * Since i*ky is constant in x, FFT_x^{-1}[i*ky * F(kx, ky)] = i*ky * FFT_x^{-1}[F(kx, ky)]!
     * Wait — what if we do the 1D y-FFT FIRST (only nkx 1D FFTs of length ny_pad!), and the 1D x-FFT SECOND? */
    for (int iky = 0; iky < nky; iky++) {
        const tgk_complex *sa = spec_a + (size_t)iky * nkx;
        const tgk_complex *sb = spec_b + (size_t)iky * nkx;
        int n_pos = (nkx > 1) ? (half_x + 1) : 1;

        memset(row_in, 0, sizeof(tgk_complex) * nx_pad);
        for (int ikx = 0; ikx < n_pos; ikx++) row_in[ikx] = sa[ikx];
        for (int ikx = half_x + 1; ikx < nkx; ikx++) row_in[nx_pad - nkx + ikx] = sa[ikx];
        fft_1d_stockham(row_in, row_out, row_tmp, nx_pad, plan->tw_x_inv, 1);
        for (int ix = 0; ix < nx_pad; ix++) scratch1[(size_t)ix * nky + iky] = row_out[ix];

        memset(row_in, 0, sizeof(tgk_complex) * nx_pad);
        for (int ikx = 0; ikx < n_pos; ikx++) row_in[ikx] = sb[ikx];
        for (int ikx = half_x + 1; ikx < nkx; ikx++) row_in[nx_pad - nkx + ikx] = sb[ikx];
        fft_1d_stockham(row_in, row_out, row_tmp, nx_pad, plan->tw_x_inv, 1);
        for (int ix = 0; ix < nx_pad; ix++) scr_b[(size_t)ix * nky + iky] = row_out[ix];
    }

    /* Step 2: For each x in [0, nx_pad-1], combine A(x, ky) + i * B(x, ky) into a single
     * 1D inverse FFT of length ny_pad, producing real_a(x, y) = Re and real_b(x, y) = Im! */
    for (int ix = 0; ix < nx_pad; ix++) {
        const tgk_complex *ca = scratch1 + (size_t)ix * nky;
        const tgk_complex *cb = scr_b    + (size_t)ix * nky;

        memset(row_in, 0, sizeof(tgk_complex) * ny_pad);
        {
            tgk_complex f0 = ca[0];
            tgk_complex f1 = cb[0];
            row_in[0].re = f0.re - f1.im;
            row_in[0].im = f0.im + f1.re;
        }
        for (int iky = 1; iky < nky; iky++) {
            tgk_complex f0 = ca[iky];
            tgk_complex f1 = cb[iky];
            row_in[iky].re          =  f0.re - f1.im;
            row_in[iky].im          =  f0.im + f1.re;
            row_in[ny_pad - iky].re =  f0.re + f1.im;
            row_in[ny_pad - iky].im = -f0.im + f1.re;
        }

        fft_1d_stockham(row_in, row_out, row_tmp, ny_pad, plan->tw_y_inv, 1);

        tgk_real *da = real_a + (size_t)ix * ny_pad;
        tgk_real *db = real_b + (size_t)ix * ny_pad;
        for (int iy = 0; iy < ny_pad; iy++) {
            da[iy] = row_out[iy].re;
            db[iy] = row_out[iy].im;
        }
    }
}

void tgk_fft_c2r_dealiased(const tgk_fft_plan *plan, const tgk_complex *spec_in,
                           tgk_real *real_out, tgk_complex *scratch1, tgk_complex *scratch2) {
    const int nkx    = plan->nkx;
    const int nky    = plan->nky;
    const int nx_pad = plan->nx_pad;
    const int ny_pad = plan->ny_pad;
    const int half_x = nkx / 2;

    /* Scratch layout:
     * scratch1 holds intermediate C[nx_pad][nky] after 1D inverse FFT along x for each ky.
     * scratch2 holds 1D FFT work buffers of length max(nx_pad, ny_pad). */
    tgk_complex *row_in  = scratch2;
    tgk_complex *row_out = scratch2 + nx_pad + ny_pad;
    tgk_complex *row_tmp = row_out  + nx_pad + ny_pad;

    /* Step 1: For each ky in [0, nky-1], zero-pad kx from nkx to nx_pad and run 1D inverse FFT in x */
    for (int iky = 0; iky < nky; iky++) {
        const tgk_complex *src_ky = spec_in + (size_t)iky * nkx;
        memset(row_in, 0, sizeof(tgk_complex) * nx_pad);

        /* Non-negative kx: 0 .. half_x */
        int n_pos = (nkx > 1) ? (half_x + 1) : 1;
        for (int ikx = 0; ikx < n_pos; ikx++) {
            row_in[ikx] = src_ky[ikx];
        }
        /* Negative kx: half_x + 1 .. nkx - 1 mapped to the end of nx_pad */
        for (int ikx = half_x + 1; ikx < nkx; ikx++) {
            row_in[nx_pad - nkx + ikx] = src_ky[ikx];
        }

        fft_1d_stockham(row_in, row_out, row_tmp, nx_pad, plan->tw_x_inv, 1);

        /* Store transposed into scratch1[ix * nky + iky] for contiguous y-transform */
        for (int ix = 0; ix < nx_pad; ix++) {
            scratch1[(size_t)ix * nky + iky] = row_out[ix];
        }
    }

    /* Step 2: For each x in [0, nx_pad-1], build Hermitian-symmetric y spectrum of length ny_pad
     * and transform to real space. Pair two x-rows at a time using Hermitian trick:
     * Z(y) = f_even(y) + i * f_odd(y) <=> Z(ky) = F_even(ky) + i * F_odd(ky) */
    for (int ix = 0; ix < nx_pad; ix += 2) {
        const tgk_complex *c0 = scratch1 + (size_t)ix * nky;
        const tgk_complex *c1 = (ix + 1 < nx_pad) ? (scratch1 + (size_t)(ix + 1) * nky) : c0;

        memset(row_in, 0, sizeof(tgk_complex) * ny_pad);

        /* ky = 0 mode: must be purely real in y for each x-row */
        {
            tgk_complex f0 = c0[0];
            tgk_complex f1 = c1[0];
            /* Note: for ky=0, reality in (x,y) implies c0[0] and c1[0] have zero imaginary part
             * if F(-kx, 0) = conj(F(kx, 0)) is satisfied, which we enforce on ky=0. */
            row_in[0].re = f0.re - f1.im;
            row_in[0].im = f0.im + f1.re;
        }

        for (int iky = 1; iky < nky; iky++) {
            tgk_complex f0 = c0[iky];
            tgk_complex f1 = c1[iky];
            /* Positive ky: F0(ky) + i * F1(ky) */
            row_in[iky].re = f0.re - f1.im;
            row_in[iky].im = f0.im + f1.re;

            /* Negative ky (ny_pad - iky): conj(F0(ky)) + i * conj(F1(ky)) */
            row_in[ny_pad - iky].re =  f0.re + f1.im;
            row_in[ny_pad - iky].im = -f0.im + f1.re;
        }

        fft_1d_stockham(row_in, row_out, row_tmp, ny_pad, plan->tw_y_inv, 1);

        tgk_real *dst0 = real_out + (size_t)ix * ny_pad;
        tgk_real *dst1 = (ix + 1 < nx_pad) ? (real_out + (size_t)(ix + 1) * ny_pad) : NULL;

        for (int iy = 0; iy < ny_pad; iy++) {
            dst0[iy] = row_out[iy].re;
            if (dst1) dst1[iy] = row_out[iy].im;
        }
    }
}

/* 2D Real-to-Complex analysis with folded 2/3-rule truncation & normalization (x-first algorithm!):
 * Input:  real_in[nx_pad][ny_pad]
 * Output: spec_out[nky][nkx]
 *
 * Key insight: By transforming along x FIRST (pairing two y-columns iy and iy+1 into a single
 * complex 1D x-FFT of length nx_pad, and immediately truncating to the nkx dealiased modes!),
 * we only need:
 *   - ny_pad / 2  1D x-FFTs of length nx_pad (48 instead of 64!)
 *   - nkx / 2     1D y-FFTs of length ny_pad (64 instead of 96!)
 * Total 1D FFTs for 2D R2C drops from 128 down to 112! */
void tgk_fft_r2c_dealiased(const tgk_fft_plan *plan, const tgk_real *real_in,
                           tgk_complex *spec_out, tgk_complex *scratch1, tgk_complex *scratch2) {
    const int nkx    = plan->nkx;
    const int nky    = plan->nky;
    const int nx_pad = plan->nx_pad;
    const int ny_pad = plan->ny_pad;
    const int half_x = nkx / 2;
    const tgk_real scale = (tgk_real)plan->norm;

    tgk_complex *row_in  = scratch2;
    tgk_complex *row_out = scratch2 + nx_pad + ny_pad;
    tgk_complex *row_tmp = row_out  + nx_pad + ny_pad;

    /* Step 1: Forward 1D FFT along x for pairs of y-columns (iy, iy+1),
     * storing only the nkx dealiased kx modes into scratch1[ikx * ny_pad + iy]! */
    for (int iy = 0; iy < ny_pad; iy += 2) {
        for (int ix = 0; ix < nx_pad; ix++) {
            const tgk_real *col = real_in + (size_t)ix * ny_pad + iy;
            row_in[ix].re = col[0];
            row_in[ix].im = (iy + 1 < ny_pad) ? col[1] : 0.0f;
        }

        fft_1d_stockham(row_in, row_out, row_tmp, nx_pad, plan->tw_x_fwd, 0);

        /* Unpack ikx = 0 */
        {
            tgk_complex z0 = row_out[0];
            scratch1[0 * ny_pad + iy]     = (tgk_complex){z0.re, 0.0f};
            if (iy + 1 < ny_pad) {
                scratch1[0 * ny_pad + iy + 1] = (tgk_complex){z0.im, 0.0f};
            }
        }
        /* Unpack pairs (+ikx, -ikx = nkx - ikx) using Hermitian symmetry of Z(kx) and Z(nx_pad - ikx) */
        for (int ikx = 1; ikx < (nkx + 1) / 2; ikx++) {
            tgk_complex zp = row_out[ikx];
            tgk_complex zm = row_out[nx_pad - ikx];
            int ikx_neg = nkx - ikx;

            tgk_complex a_pos = {0.5f * (zp.re + zm.re), 0.5f * (zp.im - zm.im)};
            tgk_complex b_pos = {0.5f * (zp.im + zm.im), 0.5f * (zm.re - zp.re)};

            scratch1[(size_t)ikx * ny_pad + iy]     = a_pos;
            scratch1[(size_t)ikx_neg * ny_pad + iy] = (tgk_complex){a_pos.re, -a_pos.im};
            if (iy + 1 < ny_pad) {
                scratch1[(size_t)ikx * ny_pad + iy + 1]     = b_pos;
                scratch1[(size_t)ikx_neg * ny_pad + iy + 1] = (tgk_complex){b_pos.re, -b_pos.im};
            }
        }
        if (nkx > 1 && (nkx & 1) == 0) {
            scratch1[(size_t)half_x * ny_pad + iy] = (tgk_complex){0.0f, 0.0f};
            if (iy + 1 < ny_pad) {
                scratch1[(size_t)half_x * ny_pad + iy + 1] = (tgk_complex){0.0f, 0.0f};
            }
        }
    }

    /* Step 2: Now notice that for each ikx in [0 .. (nkx+1)/2 - 1],
     * scratch1[ikx * ny_pad + iy] is F(+kx, y), and F(-kx, y) = conj(F(+kx, y))!
     * That means a SINGLE 1D forward y-FFT of F(+kx, y) gives BOTH:
     *   - F(+kx, +ky) = scale * row_out[iky]           (for iky = 0 .. nky-1)
     *   - F(-kx, +ky) = scale * conj(row_out[ny_pad - iky]) (for iky = 1 .. nky-1)!
     * We only need (nkx / 2) 1D y-FFTs TOTAL instead of nkx! */
    /* 2a: ikx = 0 */
    fft_1d_stockham(scratch1, row_out, row_tmp, ny_pad, plan->tw_y_fwd, 0);
    spec_out[0] = (tgk_complex){0.0f, 0.0f};
    for (int iky = 1; iky < nky; iky++) {
        spec_out[(size_t)iky * nkx + 0] = tgk_cscale(scale, row_out[iky]);
    }
    if (nkx > 1 && (nkx & 1) == 0) {
        for (int iky = 0; iky < nky; iky++) {
            spec_out[(size_t)iky * nkx + half_x] = (tgk_complex){0.0f, 0.0f};
        }
    }

    /* 2b: ikx = 1 .. (nkx+1)/2 - 1 */
    for (int ikx = 1; ikx < (nkx + 1) / 2; ikx++) {
        int ikx_neg = nkx - ikx;
        const tgk_complex *src_y = scratch1 + (size_t)ikx * ny_pad;
        fft_1d_stockham(src_y, row_out, row_tmp, ny_pad, plan->tw_y_fwd, 0);

        /* ky = 0: F(-kx, 0) = conj(F(+kx, 0)) */
        tgk_complex f0 = tgk_cscale(scale, row_out[0]);
        spec_out[ikx]     = f0;
        spec_out[ikx_neg] = (tgk_complex){f0.re, -f0.im};

        /* ky = 1 .. nky-1 */
        for (int iky = 1; iky < nky; iky++) {
            tgk_complex fp = tgk_cscale(scale, row_out[iky]);
            tgk_complex fm = tgk_cscale(scale, row_out[ny_pad - iky]);
            spec_out[(size_t)iky * nkx + ikx]     = fp;
            spec_out[(size_t)iky * nkx + ikx_neg] = (tgk_complex){fm.re, -fm.im};
        }
    }
}
