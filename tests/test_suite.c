#include "../tgk.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static int g_failures = 0;

#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "  [FAIL] " fmt "\n", ##__VA_ARGS__); \
        g_failures++; \
    } else { \
        printf("  [PASS] " fmt "\n", ##__VA_ARGS__); \
    } \
} while (0)

/* 1. Test Gauss-Laguerre quadrature exactness and custom 2/3-rule dealiased FFT round-trip */
static void test_fft_and_quadrature(void) {
    printf("\n--- Test 1: Gauss-Laguerre Quadrature, 2D Dealiased FFT & E x B Bracket ---\n");
    tgk_param p;
    tgk_param_defaults(&p);
    p.nkx = 32;
    p.nky = 16;
    p.nz  = 16;
    p.nv  = 32;
    p.nmu = 8;

    tgk_geom gm;
    tgk_grid gr;
    tgk_geom_init(&p, &gm);
    tgk_grid_init(&p, &gm, &gr);

    /* Verify Gauss-Laguerre integrates int_0^inf x^k exp(-x) dx = k! to machine precision */
    double mom0 = 0.0, mom1 = 0.0, mom2 = 0.0, mom3 = 0.0;
    for (int i = 0; i < gr.nmu; i++) {
        double x = gr.mu[i];
        double w = gr.w_mu[i];
        mom0 += w;
        mom1 += w * x;
        mom2 += w * x * x;
        mom3 += w * x * x * x;
    }
    CHECK(fabs(mom0 - 1.0) < 1e-13 && fabs(mom1 - 1.0) < 1e-12 &&
          fabs(mom2 - 2.0) < 1e-11 && fabs(mom3 - 6.0) < 1e-10,
          "Gauss-Laguerre moments (1, x, x^2, x^3): %.12f, %.12f, %.12f, %.12f",
          mom0, mom1, mom2, mom3);

    /* Test 2D c2r -> r2c round-trip accuracy and 2/3-rule dealiasing on radix-2,3,5 grid */
    tgk_fft_plan plan;
    tgk_fft_init(&plan, p.nkx, p.nky);
    size_t nplane = (size_t)p.nkx * p.nky;
    size_t nreal  = (size_t)plan.nx_pad * plan.ny_pad;

    tgk_complex *spec_in  = tgk_malloc_aligned(sizeof(tgk_complex) * nplane);
    tgk_complex *spec_out = tgk_malloc_aligned(sizeof(tgk_complex) * nplane);
    tgk_real    *real_buf = tgk_malloc_aligned(sizeof(tgk_real) * nreal);
    tgk_complex *scr1     = tgk_malloc_aligned(sizeof(tgk_complex) * (2 * nplane + plan.nx_pad * p.nky));
    tgk_complex *scr2     = tgk_malloc_aligned(sizeof(tgk_complex) * 4 * (plan.nx_pad + plan.ny_pad));

    /* Populate random Hermitian-valid spectrum */
    for (int iky = 1; iky < p.nky; iky++) {
        for (int ikx = 0; ikx < p.nkx; ikx++) {
            if (ikx == p.nkx / 2) continue;
            spec_in[iky * p.nkx + ikx].re = (tgk_real)cos(0.3 * ikx + 0.7 * iky);
            spec_in[iky * p.nkx + ikx].im = (tgk_real)sin(0.5 * ikx - 0.2 * iky);
        }
    }
    /* Zonal ky=0 reality symmetry: F(-kx, 0) = conj(F(kx, 0)) */
    for (int ikx = 1; ikx < p.nkx / 2; ikx++) {
        spec_in[ikx].re = (tgk_real)(0.5 * ikx);
        spec_in[ikx].im = (tgk_real)(-0.25 * ikx);
        spec_in[p.nkx - ikx].re =  spec_in[ikx].re;
        spec_in[p.nkx - ikx].im = -spec_in[ikx].im;
    }

    tgk_fft_c2r_dealiased(&plan, spec_in, real_buf, scr1, scr2);
    tgk_fft_r2c_dealiased(&plan, real_buf, spec_out, scr1, scr2);

    double max_err = 0.0;
    for (size_t k = 0; k < nplane; k++) {
        double err = hypot((double)spec_in[k].re - (double)spec_out[k].re,
                           (double)spec_in[k].im - (double)spec_out[k].im);
        if (err > max_err) max_err = err;
    }
    double tol = TGK_REAL_IS_FP64 ? 1e-12 : 1e-5;
    CHECK(max_err < tol, "2D Dealiased FFT round-trip max error = %.3e (pad %dx%d)",
          max_err, plan.nx_pad, plan.ny_pad);

    /* Verify Parseval's theorem between spectral and physical space */
    double e_spec = 0.0, e_real = 0.0;
    for (int iky = 0; iky < p.nky; iky++) {
        double m_fac = (iky == 0) ? 1.0 : 2.0;
        for (int ikx = 0; ikx < p.nkx; ikx++) {
            e_spec += m_fac * tgk_cabs2(spec_in[iky * p.nkx + ikx]);
        }
    }
    for (size_t r = 0; r < nreal; r++) {
        e_real += (double)real_buf[r] * (double)real_buf[r] * plan.norm;
    }
    CHECK(fabs(e_spec - e_real) / e_spec < tol,
          "2D FFT Parseval energy conservation: E_spec=%.6f, E_real=%.6f", e_spec, e_real);

    /* The dealiased bracket N = dchi/dy dh/dx - dchi/dx dh/dy must conserve sum conj(h) N and
     * sum conj(chi) N exactly, including when the even-nkx Nyquist column holds data. */
    {
        size_t nbig = 2 * nplane + 4 * (size_t)plan.nx_pad * plan.ny_pad;
        size_t nsml = 8 * (size_t)(plan.nx_pad + plan.ny_pad);
        tgk_complex *hs = tgk_malloc_aligned(sizeof(tgk_complex) * nplane);
        tgk_complex *cs = tgk_malloc_aligned(sizeof(tgk_complex) * nplane);
        tgk_complex *ns = tgk_malloc_aligned(sizeof(tgk_complex) * nplane);
        tgk_complex *b1 = tgk_malloc_aligned(sizeof(tgk_complex) * nbig);
        tgk_complex *b2 = tgk_malloc_aligned(sizeof(tgk_complex) * nsml);
        tgk_real *hx = tgk_malloc_aligned(sizeof(tgk_real) * nreal);
        tgk_real *hy = tgk_malloc_aligned(sizeof(tgk_real) * nreal);
        tgk_real *cx = tgk_malloc_aligned(sizeof(tgk_real) * nreal);
        tgk_real *cy = tgk_malloc_aligned(sizeof(tgk_real) * nreal);
        tgk_real *pr = tgk_malloc_aligned(sizeof(tgk_real) * nreal);
        memset(hs, 0, sizeof(tgk_complex) * nplane);
        memset(cs, 0, sizeof(tgk_complex) * nplane);
        for (int iky = 0; iky < p.nky; iky++) {
            for (int ikx = 0; ikx < p.nkx; ikx++) {
                if (iky == 0 && (ikx == 0 || ikx >= p.nkx / 2)) continue;
                size_t k = (size_t)iky * p.nkx + ikx;
                hs[k] = (tgk_complex){(tgk_real)cos(1.3 * ikx + 0.4 * iky), (tgk_real)sin(0.7 * ikx - 1.1 * iky)};
                cs[k] = (tgk_complex){(tgk_real)sin(0.9 * ikx + 0.3 * iky), (tgk_real)cos(0.2 * ikx + 1.7 * iky)};
            }
        }
        for (int ikx = 1; ikx < p.nkx / 2; ikx++) {
            hs[p.nkx - ikx] = (tgk_complex){hs[ikx].re, -hs[ikx].im};
            cs[p.nkx - ikx] = (tgk_complex){cs[ikx].re, -cs[ikx].im};
        }
        tgk_fft_c2r_grad_dealiased(&plan, hs, gr.kx, gr.ky, hx, hy, b1, b2);
        tgk_fft_c2r_grad_dealiased(&plan, cs, gr.kx, gr.ky, cx, cy, b1, b2);
        for (size_t r = 0; r < nreal; r++) pr[r] = cy[r] * hx[r] - cx[r] * hy[r];
        tgk_fft_r2c_dealiased(&plan, pr, ns, b1, b2);
        double ih = 0.0, ic = 0.0, nn = 0.0, hh = 0.0, cc = 0.0;
        for (int iky = 0; iky < p.nky; iky++) {
            double m = (iky == 0) ? 1.0 : 2.0;
            for (int ikx = 0; ikx < p.nkx; ikx++) {
                size_t k = (size_t)iky * p.nkx + ikx;
                ih += m * ((double)hs[k].re * ns[k].re + (double)hs[k].im * ns[k].im);
                ic += m * ((double)cs[k].re * ns[k].re + (double)cs[k].im * ns[k].im);
                nn += m * tgk_cabs2(ns[k]);
                hh += m * tgk_cabs2(hs[k]);
                cc += m * tgk_cabs2(cs[k]);
            }
        }
        double rh = fabs(ih) / sqrt(nn * hh), rc = fabs(ic) / sqrt(nn * cc);
        double btol = TGK_REAL_IS_FP64 ? 1e-13 : 1e-5;
        CHECK(rh < btol && rc < btol,
              "E x B bracket conserves <h,N> (%.1e) and <chi,N> (%.1e) with Nyquist column populated", rh, rc);
        tgk_free_aligned(hs); tgk_free_aligned(cs); tgk_free_aligned(ns);
        tgk_free_aligned(b1); tgk_free_aligned(b2);
        tgk_free_aligned(hx); tgk_free_aligned(hy); tgk_free_aligned(cx); tgk_free_aligned(cy);
        tgk_free_aligned(pr);
    }

    tgk_free_aligned(spec_in);
    tgk_free_aligned(spec_out);
    tgk_free_aligned(real_buf);
    tgk_free_aligned(scr1);
    tgk_free_aligned(scr2);
    tgk_fft_free(&plan);
    tgk_grid_free(&gr);
    tgk_geom_free(&gm);
}

/* 2. Linear Cyclone Base Case ITG growth rate and frequency vs ky benchmark
 * Reference: Dimits et al., Phys. Plasmas 7, 969 (2000) / GS2 & GENE cross-code benchmark */
/* geom: 0 = s-alpha, 1 = Miller (kappa 1.4, delta 0.2), 2 = Miller circular limit.
 * Runs to t = 40 (several e-folds at gamma ~ 0.1) and returns the late-time growth rate.
 * w_ratio_out, if given, receives W(t_end) / W(0) for the free energy W. */
static double run_linear_mode(double ky, double r_lt, double r_ln, int kinetic_elec,
                              double beta, int imex, int geom_miller,
                              double *omega_out, double *w_ratio_out) {
    tgk_param p;
    tgk_param_defaults(&p);
    p.nkx       = 8;
    p.nky       = 2;
    p.nz        = 24;
    p.nv        = 32;
    p.nmu       = 8;
    p.ky_min    = ky;
    p.kx_max    = 2.0 * 3.14159265358979323846 * p.shat * ky;
    p.nonlinear = 0;
    p.dt_max    = 0.04;
    p.nsteps    = 1000;
    p.hyper_z   = 0.20;
    p.beta      = beta;
    p.imex      = imex;

    if (geom_miller) {
        p.geom_type = TGK_GEOM_MILLER;
        p.kappa     = (geom_miller == 1) ? 1.4 : 1.0;
        p.delta     = (geom_miller == 1) ? 0.2 : 0.0;
    }

    p.spec[0].fprim = r_ln;
    p.spec[0].tprim = r_lt;

    if (kinetic_elec) {
        p.nspecies            = 2;
        p.adiabatic_electrons = 0;
        /* Reduced mass ratio m_i/m_e = 400 with IMEX enabled for rapid linear test */
        p.spec[1].mass        = 0.0025;
        p.spec[1].fprim       = r_ln;
        p.spec[1].tprim       = r_lt;
        if (!imex) p.dt_max   = 0.004;
        p.nsteps = 220;
    }

    tgk_ctx ctx;
    tgk_ctx_init(&ctx, &p);
    tgk_init_perturbation(&ctx);

    tgk_diag_out d;
    tgk_diag_compute(&ctx, ctx.h, &d);
    double w0 = d.free_energy;

    double gamma_avg = 0.0, omega_avg = 0.0;
    int navg = 0;

    for (int step = 1; step <= p.nsteps; step++) {
        tgk_step_rk4(&ctx);
        if (step % 10 == 0) {
            tgk_diag_compute(&ctx, ctx.h, &d);
            if (step >= p.nsteps - 50) {
                gamma_avg += d.gamma_max;
                omega_avg += d.omega_max;
                navg++;
            }
        }
    }

    if (w_ratio_out) *w_ratio_out = d.free_energy / w0;
    tgk_ctx_free(&ctx);
    if (omega_out) *omega_out = (navg > 0) ? (omega_avg / navg) : 0.0;
    return (navg > 0) ? (gamma_avg / navg) : 0.0;
}

static void test_linear_benchmarks(void) {
    printf("\n--- Test 2: Linear Cyclone Base Case, TEM, KBM & Miller Geometry ---\n");

    double omega_cbc3 = 0.0, omega_cbc5 = 0.0;
    double gamma_cbc3 = run_linear_mode(0.3, 6.92, 2.22, 0, 0.0, 0, 0, &omega_cbc3, NULL);
    double gamma_cbc5 = run_linear_mode(0.5, 6.92, 2.22, 0, 0.0, 0, 0, &omega_cbc5, NULL);

    /* Adiabatic-electron Cyclone Base Case, ky rho_i = 0.3: published gamma ~ 0.1 v_ti/a.
     * The real frequency is negative in TGK's sign convention (ITG direction). */
    CHECK(gamma_cbc3 > 0.07 && gamma_cbc3 < 0.13 && omega_cbc3 < 0.0,
          "Cyclone Base Case (ky=0.3) linear ITG growth rate gamma=%.4f, omega=%.4f",
          gamma_cbc3, omega_cbc3);
    /* ITG |omega| grows with ky in the same direction */
    CHECK(gamma_cbc5 > 0.0 && omega_cbc5 < omega_cbc3,
          "Cyclone Base Case (ky=0.5) unstable, |omega| increases with ky: gamma=%.4f, omega=%.4f",
          gamma_cbc5, omega_cbc5);

    /* R/L_T = 1, R/L_n = 0 is far below the ITG threshold: the free energy must decay.
     * (An instantaneous growth rate is not meaningful here: damped modes beat.) */
    double w_ratio = 0.0;
    run_linear_mode(0.3, 1.0, 0.0, 0, 0.0, 0, 0, NULL, &w_ratio);
    CHECK(w_ratio < 0.5,
          "Sub-critical gradient (R/L_T=1.0) is linearly stable: W(t=40)/W(0) = %.3f", w_ratio);

    /* Miller geometry reduces to s-alpha in the circular limit; shaped case stays unstable */
    double gamma_mcirc  = run_linear_mode(0.3, 6.92, 2.22, 0, 0.0, 0, 2, NULL, NULL);
    double gamma_miller = run_linear_mode(0.3, 6.92, 2.22, 0, 0.0, 0, 1, NULL, NULL);
    CHECK(fabs(gamma_mcirc / gamma_cbc3 - 1.0) < 0.05,
          "Miller circular limit matches s-alpha: gamma=%.4f vs %.4f", gamma_mcirc, gamma_cbc3);
    CHECK(gamma_miller > 0.02,
          "Miller shaped geometry (kappa=1.4, delta=0.2) ITG growth rate gamma=%.4f", gamma_miller);

    /* Kinetic electrons + IMEX + electromagnetic A_par (ITG/TEM/KBM capability) */
    double omega_ke = 0.0;
    double gamma_ke = run_linear_mode(0.3, 6.92, 3.50, 1, 0.005, 1, 0, &omega_ke, NULL);
    CHECK(isfinite(gamma_ke) && gamma_ke > 0.03,
          "Kinetic electrons + electromagnetic A_par (beta=0.5%%) + IMEX growth rate: gamma=%.4f, omega=%.4f",
          gamma_ke, omega_ke);

    /* Free-energy balance dW/dt = drive + hyperdiffusion: the drive term is built from the
     * heat and particle fluxes, so this checks the flux sign, the gradient normalization and the
     * mirror force against the time evolution. Linear Cyclone, diagnostics every step. */
    {
        tgk_param p;
        tgk_param_defaults(&p);
        p.nkx = 8; p.nky = 4; p.nz = 24; p.nv = 32; p.nmu = 8;
        p.nonlinear = 0;
        p.dt_max    = 0.04;
        tgk_ctx ctx;
        tgk_ctx_init(&ctx, &p);
        tgk_init_perturbation(&ctx);
        tgk_diag_out d;
        tgk_diag_compute(&ctx, ctx.h, &d);
        double sum_res = 0.0, sum_dw = 0.0, w_prev = d.free_energy;
        for (int step = 1; step <= 600; step++) {
            tgk_step_rk4(&ctx);
            tgk_diag_compute(&ctx, ctx.h, &d);
            if (step > 500) {
                sum_res += fabs(d.fe_residual);
                sum_dw  += fabs(d.free_energy - w_prev) / ctx.rk_dt;
            }
            w_prev = d.free_energy;
        }
        tgk_ctx_free(&ctx);
        double rel = sum_res / fmax(sum_dw, 1e-300);
        CHECK(rel < 0.05, "Linear free-energy balance closes: |residual| / |dW/dt| = %.4f", rel);
    }
}

/* 3. Rosenbluth-Hinton residual zonal flow
 * Prediction (k_x rho -> 0, large aspect ratio): phi(t -> inf) / phi(0) = 1 / (1 + 1.6 q^2 / sqrt(eps)).
 * The flux-surface-averaged zonal potential at the lowest kx performs weakly damped GAM
 * oscillations about the residual, so the signed value is averaged over t in [100, 200]. */
static tgk_complex64 fs_avg_zonal(const tgk_ctx *ctx, int ikx) {
    tgk_complex64 r = {0.0, 0.0};
    double wsum = 0.0;
    for (int iz = 0; iz < ctx->grid.nz; iz++) {
        size_t idx = (size_t)iz * ctx->grid.nky * ctx->grid.nkx + ikx;
        double w = ctx->geom.z_weight[iz];
        r.re += w * ctx->phi[idx].re;
        r.im += w * ctx->phi[idx].im;
        wsum += w;
    }
    r.re /= wsum;
    r.im /= wsum;
    return r;
}

static void test_rosenbluth_hinton(void) {
    printf("\n--- Test 3: Rosenbluth-Hinton Residual Zonal Flow Benchmark ---\n");
    tgk_param p;
    tgk_param_defaults(&p);
    p.nkx             = 8;
    p.nky             = 2;
    p.nz              = 24;
    p.nv              = 32;
    p.nmu             = 8;
    p.kx_max          = 0.20;
    p.nonlinear       = 0;
    p.init_zonal_only = 1;
    p.init_amp        = 1.0;
    p.spec[0].fprim   = 0.0;
    p.spec[0].tprim   = 0.0;
    p.hyper_z         = 0.05;
    p.dt_max          = 0.05;

    tgk_ctx ctx;
    tgk_ctx_init(&ctx, &p);
    tgk_init_perturbation(&ctx);

    tgk_complex64 f0 = fs_avg_zonal(&ctx, 1);
    double m0 = f0.re * f0.re + f0.im * f0.im;
    double acc = 0.0;
    int n = 0;
    while (ctx.time < 200.0) {
        tgk_step_rk4(&ctx);
        if (ctx.time >= 100.0) {
            tgk_complex64 f = fs_avg_zonal(&ctx, 1);
            acc += (f.re * f0.re + f.im * f0.im) / m0;
            n++;
        }
    }
    double residual = acc / (n > 0 ? n : 1);
    double rh = 1.0 / (1.0 + 1.6 * p.q0 * p.q0 / sqrt(p.eps));

    /* The large-aspect-ratio formula overestimates the residual at eps = 0.18; allow 30% */
    CHECK(fabs(residual / rh - 1.0) < 0.3,
          "Rosenbluth-Hinton residual <phi>/phi(0) = %.4f vs analytic %.4f (kx rho = %.3f)",
          residual, rh, ctx.grid.kx[1]);

    tgk_ctx_free(&ctx);
}

/* 4. Nonlinear Cyclone Base Case saturation
 * Small grid run to t = 200; turbulence saturates at t ~ 110. Checks that the ion heat flux
 * is outward at every sample of the saturated window, that the window is statistically
 * steady, and that chi_i is of the published order (~1-3 rho_i^2 v_ti / a). */
static void test_nonlinear_saturation(void) {
    printf("\n--- Test 4: Nonlinear Cyclone Base Case Saturation ---\n");
    tgk_param p;
    tgk_param_defaults(&p);
    p.nkx       = 16;
    p.nky       = 8;
    p.nz        = 16;
    p.nv        = 24;
    p.nmu       = 6;
    p.nonlinear = 1;
    p.init_amp  = 0.01;
    p.dt_max    = 0.05;

    tgk_ctx ctx;
    tgk_ctx_init(&ctx, &p);
    tgk_init_perturbation(&ctx);

    double q_min = 1e300, chi_sum = 0.0, q_a = 0.0, q_b = 0.0;
    int n = 0, n_a = 0, n_b = 0;
    while (ctx.time < 200.0) {
        tgk_step_rk4(&ctx);
        if (ctx.step % 5 == 0 && ctx.time >= 130.0) {
            tgk_diag_out d;
            tgk_diag_compute(&ctx, ctx.h, &d);
            if (d.heat_flux_i < q_min) q_min = d.heat_flux_i;
            chi_sum += d.chi_i;
            n++;
            if (ctx.time < 165.0) { q_a += d.heat_flux_i; n_a++; }
            else                  { q_b += d.heat_flux_i; n_b++; }
        }
    }
    tgk_ctx_free(&ctx);

    double chi = chi_sum / (n > 0 ? n : 1);
    double qa = q_a / (n_a > 0 ? n_a : 1), qb = q_b / (n_b > 0 ? n_b : 1);
    CHECK(isfinite(chi) && q_min > 0.0,
          "Saturated ion heat flux is outward at every sample: min Q_i = %.3f", q_min);
    CHECK(fabs(qa / qb - 1.0) < 0.3,
          "Saturated state is steady: <Q_i>[130,165] = %.3f, <Q_i>[165,200] = %.3f", qa, qb);
    CHECK(chi > 0.5 && chi < 6.0,
          "Nonlinear Cyclone ion diffusivity chi_i = %.3f rho_i^2 v_ti / a", chi);
}

/* 5. Collisional conservation (density, momentum, energy) & Checkpoint bit-identical restart */
static void test_collisions_and_checkpoint(void) {
    printf("\n--- Test 5: Dougherty Collisional Conservation & Bit-Identical Checkpoint Restart ---\n");
    tgk_param p;
    tgk_param_defaults(&p);
    p.nkx        = 8;
    p.nky        = 4;
    p.nz         = 12;
    p.nv         = 24;
    p.nmu        = 6;
    p.spec[0].nu = 0.05;
    p.nsteps     = 20;

    tgk_ctx ctx;
    tgk_ctx_init(&ctx, &p);
    tgk_init_perturbation(&ctx);

    /* Check discrete velocity moment conservation across a pure collision step */
    size_t k_test = 1 * p.nkx + 1; /* (iz=0, iky=1, ikx=1) */
    double n0 = 0.0, u0 = 0.0, e0 = 0.0;
    double n1 = 0.0, u1 = 0.0, e1 = 0.0;
    for (int imu = 0; imu < p.nmu; imu++) {
        for (int iv = 0; iv < p.nv; iv++) {
            size_t vidx = (imu * p.nv + iv) * p.nz + 0;
            double w = ctx.grid.w_vel[vidx];
            double v = ctx.grid.vpar[iv];
            double e = 0.5 * v * v + ctx.grid.mu[imu] * ctx.geom.bmag[0];
            size_t hidx = (imu * p.nv + iv) * ctx.grid.nfield + k_test;
            n0 += w * (double)ctx.h[hidx].re;
            u0 += w * v * (double)ctx.h[hidx].re;
            e0 += w * e * (double)ctx.h[hidx].re;
        }
    }

    tgk_coll_step_all(&ctx, ctx.h, 0.1);

    for (int imu = 0; imu < p.nmu; imu++) {
        for (int iv = 0; iv < p.nv; iv++) {
            size_t vidx = (imu * p.nv + iv) * p.nz + 0;
            double w = ctx.grid.w_vel[vidx];
            double v = ctx.grid.vpar[iv];
            double e = 0.5 * v * v + ctx.grid.mu[imu] * ctx.geom.bmag[0];
            size_t hidx = (imu * p.nv + iv) * ctx.grid.nfield + k_test;
            n1 += w * (double)ctx.h[hidx].re;
            u1 += w * v * (double)ctx.h[hidx].re;
            e1 += w * e * (double)ctx.h[hidx].re;
        }
    }
    double tol = TGK_REAL_IS_FP64 ? 1e-13 : 1e-6;
    CHECK(fabs(n1 - n0) < tol && fabs(u1 - u0) < tol && fabs(e1 - e0) < tol,
          "Dougherty collision operator conserves density (dn=%.2e), momentum (du=%.2e), energy (dE=%.2e)",
          fabs(n1 - n0), fabs(u1 - u0), fabs(e1 - e0));

    /* Advance 10 steps, write checkpoint, advance 10 more steps, then restart from checkpoint and compare */
    for (int i = 0; i < 10; i++) tgk_step_rk4(&ctx);
    const char *chk_file = "tests/test_restart.chk.bin";
    tgk_checkpoint_write(&ctx, chk_file);

    for (int i = 0; i < 10; i++) tgk_step_rk4(&ctx);

    tgk_ctx ctx2;
    tgk_ctx_init(&ctx2, &p);
    tgk_checkpoint_read(&ctx2, chk_file);
    for (int i = 0; i < 10; i++) tgk_step_rk4(&ctx2);

    int bit_identical = (memcmp(ctx.h, ctx2.h, sizeof(tgk_complex) * ctx.grid.ntot) == 0);
    CHECK(bit_identical, "Checkpoint restart is 100%% bit-identical across %zu complex state points",
          ctx.grid.ntot);

    tgk_ctx_free(&ctx);
    tgk_ctx_free(&ctx2);
}

/* 6. M4 Pro Memory Bandwidth & Fused RK Stage Benchmark at Reference Cyclone Resolution */
static void test_m4_pro_bandwidth_benchmark(void) {
    printf("\n--- Test 6: Apple M4 Pro Fused RK Stage Bandwidth Benchmark ---\n");
    tgk_param p;
    tgk_param_defaults(&p);
    /* Adiabatic-electron Cyclone Base Case resolution from Section 5:
     * nkx=128, nky=32, nz=24, nv=32, nmu=8, nspecies=1 (25.1M complex points) */
    p.nkx       = 128;
    p.nky       = 32;
    p.nz        = 24;
    p.nv        = 32;
    p.nmu       = 8;
    p.nspecies  = 1;
    p.nonlinear = 1;
    p.nsteps    = 4; /* 16 RK stages */

    tgk_ctx ctx;
    tgk_ctx_init(&ctx, &p);
    tgk_init_perturbation(&ctx);

    /* Warmup stage */
    tgk_step_rk4(&ctx);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int bench_steps = 3;
    for (int s = 0; s < bench_steps; s++) {
        tgk_step_rk4(&ctx);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
    double ms_per_step  = 1000.0 * elapsed / bench_steps;
    double ms_per_stage = 0.25 * ms_per_step;
    double bytes_h      = (double)ctx.grid.ntot * sizeof(tgk_complex);
    double gbps_h       = (5.0 * bytes_h) / (ms_per_stage * 1e6);

    CHECK(ms_per_stage > 0.0,
          "M4 Pro (%d P-cores, %s): %.2f ms/stage (%.2f ms/step) on 25.17M grid points (%.1f MB/copy, %.1f GB/s fused bandwidth)",
          ctx.nthreads, TGK_REAL_IS_FP64 ? "FP64" : "FP32",
          ms_per_stage, ms_per_step, bytes_h / (1024.0 * 1024.0), gbps_h);

    tgk_ctx_free(&ctx);
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--compare-fp64") == 0) {
        /* Same nonlinear Cyclone run in both precisions. Up to saturation onset the two
         * trajectories must agree pointwise; afterwards turbulence is chaotic and only the
         * time-averaged saturated heat flux can be compared. */
        FILE *f_in = fopen("tests/prec_check.in", "w");
        fprintf(f_in,
                "nkx = 16\nnky = 8\nnz = 16\nnv = 24\nnmu = 6\n"
                "nonlinear = 1\nnsteps = 100000\nt_max = 200\nnprint = 20\ninit_amp = 0.01\n"
                "out_prefix = tests/prec_run\n");
        fclose(f_in);

        const char *bins[2] = {"./tgk", "./tgk_fp64"};
        double p2_lin[2] = {0, 0}, qi_lin[2] = {0, 0}, qi_sat[2] = {0, 0};
        int rc[2];
        for (int b = 0; b < 2; b++) {
            char cmd[256];
            snprintf(cmd, sizeof(cmd), "%s tests/prec_check.in > /dev/null", bins[b]);
            rc[b] = system(cmd);
            FILE *fp = fopen("tests/prec_run.trace.txt", "r");
            char line[512];
            int n_sat = 0;
            while (fp && fgets(line, sizeof(line), fp)) {
                if (line[0] == '#') continue;
                int st;
                double t, dt, p2, zf, qi;
                if (sscanf(line, "%d %lf %lf %lf %lf %lf", &st, &t, &dt, &p2, &zf, &qi) != 6) continue;
                if (st == 1000) { p2_lin[b] = p2; qi_lin[b] = qi; }   /* t = 50, linear phase */
                if (t >= 130.0) { qi_sat[b] += qi; n_sat++; }
            }
            if (fp) fclose(fp);
            qi_sat[b] /= (n_sat > 0 ? n_sat : 1);
        }

        double rel_p2  = fabs(p2_lin[0] - p2_lin[1]) / fmax(fabs(p2_lin[1]), 1e-30);
        double rel_qi  = fabs(qi_lin[0] - qi_lin[1]) / fmax(fabs(qi_lin[1]), 1e-30);
        double rel_sat = fabs(qi_sat[0] - qi_sat[1]) / fmax(fabs(qi_sat[1]), 1e-30);

        CHECK(rc[0] == 0 && rc[1] == 0 && rel_p2 < 0.005 && rel_qi < 0.005,
              "FP32-mixed vs FP64, linear phase (t=50): |phi|^2 diff %.4f%%, Q_i diff %.4f%% (< 0.5%%)",
              100.0 * rel_p2, 100.0 * rel_qi);
        CHECK(rel_sat < 0.2,
              "FP32-mixed vs FP64, saturated <Q_i>[130,200]: %.3f vs %.3f (diff %.1f%% < 20%%)",
              qi_sat[0], qi_sat[1], 100.0 * rel_sat);
        return (g_failures == 0) ? 0 : 1;
    }

    printf("Running TGK Validation Suite (%s precision build)...\n",
           TGK_REAL_IS_FP64 ? "Full FP64" : "Mixed FP32/FP64");

    test_fft_and_quadrature();
    test_linear_benchmarks();
    test_rosenbluth_hinton();
    test_nonlinear_saturation();
    test_collisions_and_checkpoint();
    test_m4_pro_bandwidth_benchmark();

    if (g_failures == 0) {
        printf("\nALL VALIDATION CHECKS PASSED.\n");
        return 0;
    } else {
        fprintf(stderr, "\n%d VALIDATION CHECK(S) FAILED.\n", g_failures);
        return 1;
    }
}
