#ifndef TGK_SIMD_H
#define TGK_SIMD_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>

/* Precision selection: default FP32 state, compile with -DTGK_USE_FP64 for full FP64 */
#ifdef TGK_USE_FP64
typedef double tgk_real;
#define TGK_REAL_IS_FP64 1
#else
typedef float tgk_real;
#define TGK_REAL_IS_FP64 0
#endif

typedef struct {
    tgk_real re;
    tgk_real im;
} tgk_complex;

typedef struct {
    double re;
    double im;
} tgk_complex64;

typedef struct {
    float re;
    float im;
} tgk_complex32;

/* 128-byte cache-line alignment for Apple Silicon M4 Pro */
#define TGK_CACHELINE 128

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define TGK_HAS_NEON 1
#else
#define TGK_HAS_NEON 0
#endif

/* Helper inline complex math */
static inline tgk_complex tgk_cadd(tgk_complex a, tgk_complex b) {
    return (tgk_complex){a.re + b.re, a.im + b.im};
}

static inline tgk_complex tgk_csub(tgk_complex a, tgk_complex b) {
    return (tgk_complex){a.re - b.re, a.im - b.im};
}

static inline tgk_complex tgk_cmul(tgk_complex a, tgk_complex b) {
    return (tgk_complex){a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

/* Conj(a) * b */
static inline tgk_complex tgk_cmul_conj(tgk_complex a, tgk_complex b) {
    return (tgk_complex){a.re * b.re + a.im * b.im, a.re * b.im - a.im * b.re};
}

/* i * s * a = (-s * a.im, s * a.re) */
static inline tgk_complex tgk_cmul_is(tgk_real s, tgk_complex a) {
    return (tgk_complex){-s * a.im, s * a.re};
}

static inline tgk_complex tgk_cscale(tgk_real s, tgk_complex a) {
    return (tgk_complex){s * a.re, s * a.im};
}

static inline double tgk_cabs2(tgk_complex a) {
    return (double)a.re * (double)a.re + (double)a.im * (double)a.im;
}

#if TGK_HAS_NEON && !TGK_REAL_IS_FP64

/* 2 complex floats packed into float32x4_t: [re0, im0, re1, im1] */
typedef float32x4_t tgk_vc2;

static inline tgk_vc2 tgk_vc2_load(const tgk_complex *p) {
    return vld1q_f32((const float *)p);
}

static inline void tgk_vc2_store(tgk_complex *p, tgk_vc2 v) {
    vst1q_f32((float *)p, v);
}

static inline tgk_vc2 tgk_vc2_zero(void) {
    return vdupq_n_f32(0.0f);
}

static inline tgk_vc2 tgk_vc2_add(tgk_vc2 a, tgk_vc2 b) {
    return vaddq_f32(a, b);
}

static inline tgk_vc2 tgk_vc2_sub(tgk_vc2 a, tgk_vc2 b) {
    return vsubq_f32(a, b);
}

static inline tgk_vc2 tgk_vc2_scale(float s, tgk_vc2 a) {
    return vmulq_n_f32(a, s);
}

/* Scale each complex number by two distinct scalars s = [s0, s0, s1, s1] */
static inline tgk_vc2 tgk_vc2_mul_real2(float s0, float s1, tgk_vc2 a) {
    float32x4_t sv = {s0, s0, s1, s1};
    return vmulq_f32(a, sv);
}

static inline tgk_vc2 tgk_vc2_fma_real(tgk_vc2 acc, float s, tgk_vc2 a) {
    return vfmaq_n_f32(acc, a, s);
}

/* Multiply by i * [s0, s1]: i*(re + i*im) = (-im + i*re) */
static inline tgk_vc2 tgk_vc2_mul_is2(float s0, float s1, tgk_vc2 a) {
#if defined(__ARM_FEATURE_COMPLEX)
    float32x4_t sv = {s0, 0.0f, s1, 0.0f};
    float32x4_t zero = vdupq_n_f32(0.0f);
    return vcmlaq_rot90_f32(zero, sv, a);
#else
    float32x4_t rev = vrev64q_f32(a); /* [im0, re0, im1, re1] */
    float32x4_t sv = {-s0, s0, -s1, s1};
    return vmulq_f32(rev, sv);
#endif
}

static inline tgk_vc2 tgk_vc2_fma_is2(tgk_vc2 acc, float s0, float s1, tgk_vc2 a) {
#if defined(__ARM_FEATURE_COMPLEX)
    float32x4_t sv = {s0, 0.0f, s1, 0.0f};
    return vcmlaq_rot90_f32(acc, sv, a);
#else
    float32x4_t rev = vrev64q_f32(a);
    float32x4_t sv = {-s0, s0, -s1, s1};
    return vfmaq_f32(acc, rev, sv);
#endif
}

/* Complex multiply a * b for 2 complex numbers using ARMv8.3-A FCMLA if available */
static inline tgk_vc2 tgk_vc2_mul(tgk_vc2 a, tgk_vc2 b) {
#if defined(__ARM_FEATURE_COMPLEX)
    float32x4_t res = vdupq_n_f32(0.0f);
    res = vcmlaq_f32(res, a, b);
    res = vcmlaq_rot90_f32(res, a, b);
    return res;
#else
    float32x4_t a_re = vtrn1q_f32(a, a);
    float32x4_t a_im = vtrn2q_f32(a, a);
    float32x4_t b_rev = vrev64q_f32(b);
    float32x4_t prod1 = vmulq_f32(a_re, b);
    float32x4_t prod2 = vmulq_f32(a_im, b_rev);
    const float32x4_t signs = {-1.0f, 1.0f, -1.0f, 1.0f};
    return vfmaq_f32(prod1, prod2, signs);
#endif
}

static inline tgk_vc2 tgk_vc2_fma(tgk_vc2 acc, tgk_vc2 a, tgk_vc2 b) {
#if defined(__ARM_FEATURE_COMPLEX)
    acc = vcmlaq_f32(acc, a, b);
    acc = vcmlaq_rot90_f32(acc, a, b);
    return acc;
#else
    return vaddq_f32(acc, tgk_vc2_mul(a, b));
#endif
}

#endif /* TGK_HAS_NEON && !TGK_REAL_IS_FP64 */

/* Pairwise FP64 summation for accuracy in velocity-space integrals and diagnostics */
static inline double tgk_pairwise_sum_f64(const double *data, size_t n) {
    if (n == 0) return 0.0;
    if (n <= 8) {
        double s = 0.0;
        for (size_t i = 0; i < n; i++) s += data[i];
        return s;
    }
    size_t mid = n / 2;
    return tgk_pairwise_sum_f64(data, mid) + tgk_pairwise_sum_f64(data + mid, n - mid);
}

#endif /* TGK_SIMD_H */
