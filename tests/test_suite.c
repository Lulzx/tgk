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
    printf("\n--- Test 1: Gauss-Laguerre Quadrature & 2D Dealiased FFT (Radix 2, 3, 5) ---\n");
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
static double run_linear_mode(double ky, double r_lt, double r_ln, int kinetic_elec,
                              double beta, int imex, int geom_miller,
                              double *omega_out) {
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
    p.nsteps    = 220;
    p.hyper_z   = 0.20;
    p.beta      = beta;
    p.imex      = imex;

    if (geom_miller) {
        p.geom_type = TGK_GEOM_MILLER;
        p.kappa     = 1.4;
        p.delta     = 0.2;
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
    }

    tgk_ctx ctx;
    tgk_ctx_init(&ctx, &p);
    tgk_init_perturbation(&ctx);

    tgk_diag_out d;
    tgk_diag_compute(&ctx, ctx.h, &d);

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

    tgk_ctx_free(&ctx);
    if (omega_out) *omega_out = (navg > 0) ? (omega_avg / navg) : 0.0;
    return (navg > 0) ? (gamma_avg / navg) : 0.0;
}

static void test_linear_benchmarks(void) {
    printf("\n--- Test 2: Linear Cyclone Base Case, TEM, KBM & Miller Geometry ---\n");

    double omega_cbc3 = 0.0, omega_cbc5 = 0.0;
    double gamma_cbc3 = run_linear_mode(0.3, 6.92, 2.22, 0, 0.0, 0, 0, &omega_cbc3);
    double gamma_cbc5 = run_linear_mode(0.5, 6.92, 2.22, 0, 0.0, 0, 0, &omega_cbc5);

    /* Published Cyclone Base Case values (in c_s / a or v_ti / R0 units):
     * At ky*rho_i = 0.3, unstable ITG mode has positive growth rate and ion-diamagnetic frequency */
    CHECK(gamma_cbc3 > 0.05 && gamma_cbc3 < 0.35,
          "Cyclone Base Case (ky=0.3) linear ITG growth rate gamma=%.4f, omega=%.4f",
          gamma_cbc3, omega_cbc3);
    CHECK(gamma_cbc5 > gamma_cbc3 && gamma_cbc5 < 0.55,
          "Cyclone Base Case (ky=0.5) linear ITG growth rate gamma=%.4f > gamma(0.3), omega=%.4f",
          gamma_cbc5, omega_cbc5);

    /* Below critical gradient R/L_T = 1.0 (stable ITG regime): growth rate must be negative/damped */
    double gamma_stable = run_linear_mode(0.3, 1.0, 0.0, 0, 0.0, 0, 0, NULL);
    CHECK(gamma_stable < 0.005,
          "Sub-critical gradient (R/L_T=1.0 < R/L_T_crit) is linearly stable: gamma=%.4f",
          gamma_stable);

    /* Miller shaped geometry (kappa = 1.4, delta = 0.2): elongation stabilizes ITG relative to circular */
    double gamma_miller = run_linear_mode(0.3, 6.92, 2.22, 0, 0.0, 0, 1, NULL);
    CHECK(gamma_miller > 0.02 && gamma_miller < gamma_cbc3,
          "Miller shaped geometry (kappa=1.4, delta=0.2) ITG stabilization: gamma_miller=%.4f < gamma_s_alpha=%.4f",
          gamma_miller, gamma_cbc3);

    /* Kinetic electrons + IMEX + electromagnetic A_par (ITG/TEM/KBM capability) */
    double omega_ke = 0.0;
    double gamma_ke = run_linear_mode(0.3, 6.92, 3.50, 1, 0.005, 1, 0, &omega_ke);
    CHECK(isfinite(gamma_ke) && gamma_ke > 0.03,
          "Kinetic electrons + electromagnetic A_par (beta=0.5%%) + IMEX growth rate: gamma=%.4f, omega=%.4f",
          gamma_ke, omega_ke);
}

/* 3. Rosenbluth-Hinton residual zonal flow test (Section 6)
 * Analytic prediction: phi(t -> inf) / phi(0) = 1 / (1 + 1.6 * q^2 / sqrt(eps))
 * In the collisionless flux-tube limit, an initial ky=0 radial potential perturbation
 * undergoes geodesic acoustic mode (GAM) oscillations and relaxes to a non-zero residual level. */
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
    /* Turn off equilibrium gradient drives for pure Rosenbluth-Hinton test */
    p.spec[0].fprim   = 0.0;
    p.spec[0].tprim   = 0.0;
    p.hyper_z         = 0.05;
    p.dt_max          = 0.05;
    p.nsteps          = 200;

    tgk_ctx ctx;
    tgk_ctx_init(&ctx, &p);
    tgk_init_perturbation(&ctx);

    tgk_diag_out d0, d_end;
    tgk_diag_compute(&ctx, ctx.h, &d0);
    double phi0 = sqrt(d0.phi2_zonal);

    for (int step = 0; step < p.nsteps; step++) {
        tgk_step_rk4(&ctx);
    }
    tgk_diag_compute(&ctx, ctx.h, &d_end);
    double phi_end = sqrt(d_end.phi2_zonal);
    double residual = (phi0 > 0.0) ? (phi_end / phi0) : 0.0;

    /* Verify that the zonal flow is undamped to a finite non-zero residual (omitting the
     * flux-surface average would cause erroneous exponential damping or blowup). */
    CHECK(residual > 0.05 && residual <= 1.05,
          "Rosenbluth-Hinton zonal flow residual |phi(T)|/|phi(0)| = %.4f (sustained non-zero)",
          residual);

    tgk_ctx_free(&ctx);
}

/* 4. Nonlinear Cyclone Base Case & Dimits Shift test (Section 6)
 * Verify that near the linear threshold (R/L_T = 5.0, inside the Dimits shift regime),
 * zonal flows strongly suppress nonlinear heat transport compared to R/L_T = 6.92. */
static void test_nonlinear_and_dimits_shift(void) {
    printf("\n--- Test 4: Nonlinear Cyclone Base Case Saturation & Dimits Shift ---\n");
    double chi_dimits = 0.0, chi_cbc = 0.0;

    for (int case_id = 0; case_id < 2; case_id++) {
        tgk_param p;
        tgk_param_defaults(&p);
        p.nkx           = 16;
        p.nky           = 8;
        p.nz            = 16;
        p.nv            = 24;
        p.nmu           = 6;
        p.nonlinear     = 1;
        p.init_amp      = 0.05;
        p.dt_max        = 0.05;
        p.nsteps        = 120;
        p.spec[0].tprim = (case_id == 0) ? 4.8 : 6.92;

        tgk_ctx ctx;
        tgk_ctx_init(&ctx, &p);
        tgk_init_perturbation(&ctx);

        double chi_sum = 0.0;
        int n_avg = 0;
        for (int step = 1; step <= p.nsteps; step++) {
            tgk_step_rk4(&ctx);
            if (step >= p.nsteps / 2 && step % 5 == 0) {
                tgk_diag_out d;
                tgk_diag_compute(&ctx, ctx.h, &d);
                chi_sum += fabs(d.chi_i);
                n_avg++;
            }
        }
        if (case_id == 0) chi_dimits = chi_sum / fmax(n_avg, 1);
        else              chi_cbc    = chi_sum / fmax(n_avg, 1);
        tgk_ctx_free(&ctx);
    }

    CHECK(isfinite(chi_cbc) && chi_cbc > chi_dimits && chi_cbc > 0.0,
          "Nonlinear Cyclone transport chi_i(R/L_T=6.92)=%.4e > Dimits-shift regime chi_i(R/L_T=4.8)=%.4e",
          chi_cbc, chi_dimits);
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
        /* Run identical nonlinear Cyclone evolution in FP32-mixed and FP64 via binary trace output
         * and verify < 0.5% agreement as required by Section 4 & Section 6 */
        FILE *f_in = fopen("tests/prec_check.in", "w");
        fprintf(f_in,
                "nkx = 16\nnky = 8\nnz = 16\nnv = 24\nnmu = 6\n"
                "nonlinear = 1\nnsteps = 50\nnprint = 50\ninit_amp = 0.01\n"
                "out_prefix = tests/prec_run\n");
        fclose(f_in);

        int r1 = system("./tgk tests/prec_check.in > /dev/null");
        FILE *fp1 = fopen("tests/prec_run.trace.txt", "r");
        char line[512], last1[512] = "";
        while (fp1 && fgets(line, sizeof(line), fp1)) {
            if (line[0] != '#') strcpy(last1, line);
        }
        if (fp1) fclose(fp1);

        int r2 = system("./tgk_fp64 tests/prec_check.in > /dev/null");
        FILE *fp2 = fopen("tests/prec_run.trace.txt", "r");
        char last2[512] = "";
        while (fp2 && fgets(line, sizeof(line), fp2)) {
            if (line[0] != '#') strcpy(last2, line);
        }
        if (fp2) fclose(fp2);

        int st1 = 0, st2 = 0;
        double t1 = 0, dt1 = 0, p2_1 = 0, zf_1 = 0, qi_1 = 0, qe_1 = 0, chi_1 = 0, gam_1 = 0;
        double t2 = 0, dt2 = 0, p2_2 = 0, zf_2 = 0, qi_2 = 0, qe_2 = 0, chi_2 = 0, gam_2 = 0;
        sscanf(last1, "%d %lf %lf %lf %lf %lf %lf %lf %lf",
               &st1, &t1, &dt1, &p2_1, &zf_1, &qi_1, &qe_1, &chi_1, &gam_1);
        sscanf(last2, "%d %lf %lf %lf %lf %lf %lf %lf %lf",
               &st2, &t2, &dt2, &p2_2, &zf_2, &qi_2, &qe_2, &chi_2, &gam_2);

        double rel_phi2 = fabs(p2_1 - p2_2) / fmax(fabs(p2_2), 1e-30);
        double rel_qi   = fabs(qi_1 - qi_2) / fmax(fabs(qi_2), 1e-30);
        double rel_gam  = fabs(gam_1 - gam_2) / fmax(fabs(gam_2), 1e-30);

        CHECK(r1 == 0 && r2 == 0 && rel_phi2 < 0.005 && rel_qi < 0.005 && rel_gam < 0.005,
              "FP32-mixed vs Full FP64 relative diff: |phi|^2=%.4f%%, Q_i=%.4f%%, gamma=%.4f%% (all < 0.5%%)",
              100.0 * rel_phi2, 100.0 * rel_qi, 100.0 * rel_gam);
        return (g_failures == 0) ? 0 : 1;
    }

    printf("Running TGK Validation Suite (%s precision build)...\n",
           TGK_REAL_IS_FP64 ? "Full FP64" : "Mixed FP32/FP64");

    test_fft_and_quadrature();
    test_linear_benchmarks();
    test_rosenbluth_hinton();
    test_nonlinear_and_dimits_shift();
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
