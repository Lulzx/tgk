#include "tgk.h"
#include <stdlib.h>
#include <string.h>

/* Linearized Dougherty (Lenard-Bernstein) collision operator with exact discrete
 * number, parallel momentum, and kinetic energy conservation:
 *
 * C[h_s] = nu_s * [ d/dv_par ( v_par h_s + dh_s/dv_par )
 *                 + 2 * d/dmu ( mu * (h_s + (1/B) dh_s/dmu) )
 *                 + ( u_s * v_par + q_s * (E_s - 3/2) ) * F_0s ]
 *
 * Because the discrete velocity grid (nv, nmu) has finite extent and quadrature weights w_vel,
 * we first apply the diffusive/drag step in v_par and mu, and then project the change delta_h
 * so that the discrete velocity moments (1, v_par, E = v_par^2/2 + mu*B) with weight w_vel
 * are conserved to machine precision at every (kx, ky, z). */

static void solve_tridiag_complex(int n, const double *a, const double *b, const double *c,
                                  tgk_complex *rhs, double *cp_scratch) {
    double inv_b0 = 1.0 / b[0];
    cp_scratch[0] = c[0] * inv_b0;
    rhs[0].re = (tgk_real)((double)rhs[0].re * inv_b0);
    rhs[0].im = (tgk_real)((double)rhs[0].im * inv_b0);

    for (int i = 1; i < n; i++) {
        double denom = 1.0 / (b[i] - a[i] * cp_scratch[i - 1]);
        cp_scratch[i] = c[i] * denom;
        rhs[i].re = (tgk_real)(((double)rhs[i].re - a[i] * (double)rhs[i - 1].re) * denom);
        rhs[i].im = (tgk_real)(((double)rhs[i].im - a[i] * (double)rhs[i - 1].im) * denom);
    }

    for (int i = n - 2; i >= 0; i--) {
        rhs[i].re = (tgk_real)((double)rhs[i].re - cp_scratch[i] * (double)rhs[i + 1].re);
        rhs[i].im = (tgk_real)((double)rhs[i].im - cp_scratch[i] * (double)rhs[i + 1].im);
    }
}

void tgk_coll_step_all(tgk_ctx *ctx, tgk_complex *h_inout, double dt) {
    const tgk_grid  *gr = &ctx->grid;
    const tgk_geom  *gm = &ctx->geom;
    const tgk_param *p  = &ctx->param;

    const int nz       = gr->nz;
    const int nv       = gr->nv;
    const int nmu      = gr->nmu;
    const int nspecies = gr->nspecies;
    const size_t nplane = gr->nplane;
    const size_t nfield = gr->nfield;

    /* Check if any species has non-zero collision frequency */
    int any_coll = 0;
    for (int s = 0; s < nspecies; s++) {
        if (p->spec[s].nu > 1e-14) any_coll = 1;
    }
    if (!any_coll) return;

    double *a_v  = calloc(nv, sizeof(double));
    double *b_v  = calloc(nv, sizeof(double));
    double *c_v  = calloc(nv, sizeof(double));
    double *cp_v = calloc(nv, sizeof(double));
    tgk_complex *col_v = calloc(nv, sizeof(tgk_complex));

    double inv_dv2 = 1.0 / (gr->dv * gr->dv);
    double inv_2dv = 1.0 / (2.0 * gr->dv);

    for (int s = 0; s < nspecies; s++) {
        double nu = p->spec[s].nu;
        if (nu <= 1e-14) continue;
        double nudt = nu * dt;

        /* Build implicit backward-Euler tridiagonal matrix in v_parallel:
         * (I - nu*dt * L_v) h^{*} = h^n
         * where L_v h = d^2h/dv^2 + v dh/dv + h */
        for (int iv = 0; iv < nv; iv++) {
            double v = gr->vpar[iv];
            if (iv == 0 || iv == nv - 1) {
                a_v[iv] = 0.0;
                b_v[iv] = 1.0 + nudt * (2.0 * inv_dv2);
                c_v[iv] = 0.0;
            } else {
                a_v[iv] = -nudt * (inv_dv2 - v * inv_2dv);
                b_v[iv] =  1.0 - nudt * (-2.0 * inv_dv2 + 1.0);
                c_v[iv] = -nudt * (inv_dv2 + v * inv_2dv);
            }
        }

        for (int iz = 0; iz < nz; iz++) {
            double b = gm->bmag[iz];
            size_t z_off = (size_t)iz * nplane;

            /* Precompute discrete moment basis inner products <F0 * phi_a * phi_b> at (s, iz)
             * where phi_0 = 1, phi_1 = v_par, phi_2 = (v_par^2/2 + mu*B - <E>_0) */
            double m00 = 0.0, m11 = 0.0, e_mean = 0.0, m22 = 0.0;
            for (int imu = 0; imu < nmu; imu++) {
                double mu = gr->mu[imu];
                for (int iv = 0; iv < nv; iv++) {
                    double v = gr->vpar[iv];
                    size_t vidx = (((size_t)s * nmu + imu) * nv + iv) * nz + iz;
                    double wf0 = gr->w_vel[vidx] * (double)gr->f0[vidx];
                    double e = 0.5 * v * v + mu * b;
                    m00    += wf0;
                    m11    += wf0 * v * v;
                    e_mean += wf0 * e;
                }
            }
            e_mean /= m00;
            for (int imu = 0; imu < nmu; imu++) {
                double mu = gr->mu[imu];
                for (int iv = 0; iv < nv; iv++) {
                    double v = gr->vpar[iv];
                    size_t vidx = (((size_t)s * nmu + imu) * nv + iv) * nz + iz;
                    double wf0 = gr->w_vel[vidx] * (double)gr->f0[vidx];
                    double de = (0.5 * v * v + mu * b) - e_mean;
                    m22 += wf0 * de * de;
                }
            }

            /* Loop over Fourier modes (ky, kx) on this z-plane */
            for (size_t k = 0; k < nplane; k++) {
                if (k == 0) continue; /* skip (0,0) */

                /* 1. Compute initial moments (n0, u0, T0) before collisions */
                tgk_complex64 n_init = {0.0, 0.0}, u_init = {0.0, 0.0}, t_init = {0.0, 0.0};
                for (int imu = 0; imu < nmu; imu++) {
                    double mu = gr->mu[imu];
                    for (int iv = 0; iv < nv; iv++) {
                        double v = gr->vpar[iv];
                        double de = (0.5 * v * v + mu * b) - e_mean;
                        size_t vidx = (((size_t)s * nmu + imu) * nv + iv) * nz + iz;
                        double w = gr->w_vel[vidx];
                        size_t hidx = (((size_t)s * nmu + imu) * nv + iv) * nfield + z_off + k;
                        tgk_complex hval = h_inout[hidx];

                        n_init.re += w * (double)hval.re;
                        n_init.im += w * (double)hval.im;
                        u_init.re += w * v * (double)hval.re;
                        u_init.im += w * v * (double)hval.im;
                        t_init.re += w * de * (double)hval.re;
                        t_init.im += w * de * (double)hval.im;
                    }
                }

                /* 2. Apply implicit v_parallel diffusion & drag for each mu, plus mu pitch-angle damping */
                for (int imu = 0; imu < nmu; imu++) {
                    double mu_damp = 1.0 / (1.0 + 2.0 * nudt * gr->mu[imu] * b);
                    for (int iv = 0; iv < nv; iv++) {
                        size_t hidx = (((size_t)s * nmu + imu) * nv + iv) * nfield + z_off + k;
                        col_v[iv] = tgk_cscale((tgk_real)mu_damp, h_inout[hidx]);
                    }
                    solve_tridiag_complex(nv, a_v, b_v, c_v, col_v, cp_v);
                    for (int iv = 0; iv < nv; iv++) {
                        size_t hidx = (((size_t)s * nmu + imu) * nv + iv) * nfield + z_off + k;
                        h_inout[hidx] = col_v[iv];
                    }
                }

                /* 3. Compute post-diffusion moments and restore exact conservation of (n, u_par, E) */
                tgk_complex64 n_post = {0.0, 0.0}, u_post = {0.0, 0.0}, t_post = {0.0, 0.0};
                for (int imu = 0; imu < nmu; imu++) {
                    double mu = gr->mu[imu];
                    for (int iv = 0; iv < nv; iv++) {
                        double v = gr->vpar[iv];
                        double de = (0.5 * v * v + mu * b) - e_mean;
                        size_t vidx = (((size_t)s * nmu + imu) * nv + iv) * nz + iz;
                        double w = gr->w_vel[vidx];
                        size_t hidx = (((size_t)s * nmu + imu) * nv + iv) * nfield + z_off + k;
                        tgk_complex hval = h_inout[hidx];

                        n_post.re += w * (double)hval.re;
                        n_post.im += w * (double)hval.im;
                        u_post.re += w * v * (double)hval.re;
                        u_post.im += w * v * (double)hval.im;
                        t_post.re += w * de * (double)hval.re;
                        t_post.im += w * de * (double)hval.im;
                    }
                }

                tgk_complex64 dn = {(n_init.re - n_post.re) / m00, (n_init.im - n_post.im) / m00};
                tgk_complex64 du = {(u_init.re - u_post.re) / m11, (u_init.im - u_post.im) / m11};
                tgk_complex64 dt_e = {(t_init.re - t_post.re) / m22, (t_init.im - t_post.im) / m22};

                for (int imu = 0; imu < nmu; imu++) {
                    double mu = gr->mu[imu];
                    for (int iv = 0; iv < nv; iv++) {
                        double v = gr->vpar[iv];
                        double de = (0.5 * v * v + mu * b) - e_mean;
                        size_t vidx = (((size_t)s * nmu + imu) * nv + iv) * nz + iz;
                        double f0 = (double)gr->f0[vidx];
                        size_t hidx = (((size_t)s * nmu + imu) * nv + iv) * nfield + z_off + k;

                        double corr_re = f0 * (dn.re + v * du.re + de * dt_e.re);
                        double corr_im = f0 * (dn.im + v * du.im + de * dt_e.im);
                        h_inout[hidx].re = (tgk_real)((double)h_inout[hidx].re + corr_re);
                        h_inout[hidx].im = (tgk_real)((double)h_inout[hidx].im + corr_im);
                    }
                }
            }
        }
    }

    free(a_v);
    free(b_v);
    free(c_v);
    free(cp_v);
    free(col_v);
}
