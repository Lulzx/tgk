#include "tgk.h"
#include <stdlib.h>
#include <string.h>

/* IMEX implicit solver for stiff parallel electron streaming along twist-and-shift linked
 * ballooning chains in z.
 *
 * When kinetic electrons are present, v_th_e / v_th_i = sqrt(m_i / m_e) ~ 60.
 * The implicit operator solves:
 *   (I + 0.5 * dt * v_th_e * v_par * b_hat.grad(z) * d/dz) h^{n+1}
 *   = (I - 0.5 * dt * v_th_e * v_par * b_hat.grad(z) * d/dz) h^{*}
 * along each linked (kx_chain, ky, mu, v_par) 1D chain across z. */

int tgk_imex_init(tgk_ctx *ctx) {
    ctx->imex.dt_cached   = 0.0;
    ctx->imex.initialized = 1;
    return 0;
}

void tgk_imex_free(tgk_ctx *ctx) {
    ctx->imex.initialized = 0;
}

/* Solve cyclic tridiagonal system for ky=0 (periodic in z) via Sherman-Morrison formula */
static void solve_cyclic_tridiag(int n, const double *a, const double *b, const double *c,
                                 tgk_complex *x, double *u, double *z_vec, double *cp) {
    double gamma = -b[0];
    double b0_mod = b[0] - gamma;
    double bn_mod = b[n - 1] - (a[0] * c[n - 1]) / gamma;

    /* Forward sweep for x and u */
    double inv_b = 1.0 / b0_mod;
    cp[0]    = c[0] * inv_b;
    x[0].re  = (tgk_real)((double)x[0].re * inv_b);
    x[0].im  = (tgk_real)((double)x[0].im * inv_b);
    z_vec[0] = gamma * inv_b;

    for (int i = 1; i < n; i++) {
        double bi = (i == n - 1) ? bn_mod : b[i];
        double ui = (i == n - 1) ? c[n - 1] : 0.0;
        double denom = 1.0 / (bi - a[i] * cp[i - 1]);
        cp[i]    = c[i] * denom;
        x[i].re  = (tgk_real)(((double)x[i].re - a[i] * (double)x[i - 1].re) * denom);
        x[i].im  = (tgk_real)(((double)x[i].im - a[i] * (double)x[i - 1].im) * denom);
        z_vec[i] = (ui - a[i] * z_vec[i - 1]) * denom;
    }

    /* Backward sweep */
    for (int i = n - 2; i >= 0; i--) {
        x[i].re  = (tgk_real)((double)x[i].re - cp[i] * (double)x[i + 1].re);
        x[i].im  = (tgk_real)((double)x[i].im - cp[i] * (double)x[i + 1].im);
        z_vec[i] = z_vec[i] - cp[i] * z_vec[i + 1];
    }

    /* Sherman-Morrison rank-1 correction: v = [1, 0, ..., 0, a[0]/gamma] */
    double v_last = a[0] / gamma;
    double vx_re  = (double)x[0].re + v_last * (double)x[n - 1].re;
    double vx_im  = (double)x[0].im + v_last * (double)x[n - 1].im;
    double vz     = 1.0 + z_vec[0] + v_last * z_vec[n - 1];
    double fac_re = vx_re / vz;
    double fac_im = vx_im / vz;

    for (int i = 0; i < n; i++) {
        x[i].re = (tgk_real)((double)x[i].re - fac_re * z_vec[i]);
        x[i].im = (tgk_real)((double)x[i].im - fac_im * z_vec[i]);
    }
}

void tgk_imex_solve_electrons(tgk_ctx *ctx, tgk_complex *h_inout, double dt) {
    const tgk_grid  *gr = &ctx->grid;
    const tgk_geom  *gm = &ctx->geom;
    const tgk_param *p  = &ctx->param;
    if (!p->imex) return;

    const int nkx      = gr->nkx;
    const int nky      = gr->nky;
    const int nz       = gr->nz;
    const int nv       = gr->nv;
    const int nmu      = gr->nmu;
    const int nspecies = gr->nspecies;
    const size_t nfield = gr->nfield;

    int max_chain = nkx * nz;
    double      *a_ch  = calloc(max_chain, sizeof(double));
    double      *b_ch  = calloc(max_chain, sizeof(double));
    double      *c_ch  = calloc(max_chain, sizeof(double));
    double      *cp_ch = calloc(max_chain, sizeof(double));
    double      *z_ch  = calloc(max_chain, sizeof(double));
    tgk_complex *rhs_c = calloc(max_chain, sizeof(tgk_complex));
    tgk_complex *u_in  = calloc(max_chain, sizeof(tgk_complex));
    int         *kx_ch = calloc(nkx, sizeof(int));
    int         *vis   = calloc(nkx, sizeof(int));

    for (int s = 0; s < nspecies; s++) {
        if (p->spec[s].charge >= 0.0) continue; /* only kinetic electrons */
        double vthe = sqrt(p->spec[s].temp / p->spec[s].mass);

        for (int imu = 0; imu < nmu; imu++) {
            for (int iv = 0; iv < nv; iv++) {
                double vpar = gr->vpar[iv];
                if (fabs(vpar) < 1e-12) continue;

                size_t slab_off = (((size_t)s * nmu + imu) * nv + iv) * nfield;
                tgk_complex *h_slab = h_inout + slab_off;

                for (int iky = 0; iky < nky; iky++) {
                    memset(vis, 0, sizeof(int) * nkx);

                    for (int ikx_start = 0; ikx_start < nkx; ikx_start++) {
                        if (vis[ikx_start]) continue;
                        if (iky == 0 && ikx_start == 0) {
                            vis[0] = 1;
                            continue;
                        }

                        /* Check if this mode is periodic or open twist-and-shift chain */
                        int is_periodic = (gr->ts_minus[(size_t)iky * nkx + ikx_start] == ikx_start);

                        if (is_periodic) {
                            vis[ikx_start] = 1;
                            for (int iz = 0; iz < nz; iz++) {
                                double alpha = 0.25 * dt * vthe * vpar * gm->gradpar[iz] / gm->dz;
                                a_ch[iz] = -alpha;
                                b_ch[iz] =  1.0;
                                c_ch[iz] =  alpha;
                                u_in[iz] = h_slab[((size_t)iz * nky + iky) * nkx + ikx_start];
                            }
                            for (int iz = 0; iz < nz; iz++) {
                                int izm1 = (iz - 1 + nz) % nz;
                                int izp1 = (iz + 1) % nz;
                                double alpha = c_ch[iz];
                                rhs_c[iz].re = (tgk_real)((double)u_in[iz].re - alpha * ((double)u_in[izp1].re - (double)u_in[izm1].re));
                                rhs_c[iz].im = (tgk_real)((double)u_in[iz].im - alpha * ((double)u_in[izp1].im - (double)u_in[izm1].im));
                            }
                            solve_cyclic_tridiag(nz, a_ch, b_ch, c_ch, rhs_c, NULL, z_ch, cp_ch);
                            for (int iz = 0; iz < nz; iz++) {
                                h_slab[((size_t)iz * nky + iky) * nkx + ikx_start] = rhs_c[iz];
                            }
                        } else {
                            /* Rewind to left-most root of the twist-and-shift chain */
                            int curr = ikx_start;
                            while (1) {
                                int prev = gr->ts_minus[(size_t)iky * nkx + curr];
                                if (prev < 0 || vis[prev]) break;
                                curr = prev;
                            }
                            int n_links = 0;
                            while (curr >= 0 && !vis[curr] && n_links < nkx) {
                                vis[curr] = 1;
                                kx_ch[n_links++] = curr;
                                curr = gr->ts_plus[(size_t)iky * nkx + curr];
                            }

                            int n_tot = n_links * nz;
                            for (int l = 0; l < n_links; l++) {
                                int ikx = kx_ch[l];
                                for (int iz = 0; iz < nz; iz++) {
                                    int m = l * nz + iz;
                                    double alpha = 0.25 * dt * vthe * vpar * gm->gradpar[iz] / gm->dz;
                                    a_ch[m] = (m > 0)         ? -alpha : 0.0;
                                    b_ch[m] = 1.0;
                                    c_ch[m] = (m + 1 < n_tot) ?  alpha : 0.0;
                                    u_in[m] = h_slab[((size_t)iz * nky + iky) * nkx + ikx];
                                }
                            }
                            for (int m = 0; m < n_tot; m++) {
                                tgk_complex um1 = (m > 0)         ? u_in[m - 1] : (tgk_complex){0.0f, 0.0f};
                                tgk_complex up1 = (m + 1 < n_tot) ? u_in[m + 1] : (tgk_complex){0.0f, 0.0f};
                                double alpha = (m + 1 < n_tot) ? c_ch[m] : -a_ch[m];
                                rhs_c[m].re = (tgk_real)((double)u_in[m].re - alpha * ((double)up1.re - (double)um1.re));
                                rhs_c[m].im = (tgk_real)((double)u_in[m].im - alpha * ((double)up1.im - (double)um1.im));
                            }

                            /* Standard Thomas solve along the open chain */
                            double inv_b0 = 1.0 / b_ch[0];
                            cp_ch[0] = c_ch[0] * inv_b0;
                            rhs_c[0].re = (tgk_real)((double)rhs_c[0].re * inv_b0);
                            rhs_c[0].im = (tgk_real)((double)rhs_c[0].im * inv_b0);
                            for (int m = 1; m < n_tot; m++) {
                                double denom = 1.0 / (b_ch[m] - a_ch[m] * cp_ch[m - 1]);
                                cp_ch[m] = c_ch[m] * denom;
                                rhs_c[m].re = (tgk_real)(((double)rhs_c[m].re - a_ch[m] * (double)rhs_c[m - 1].re) * denom);
                                rhs_c[m].im = (tgk_real)(((double)rhs_c[m].im - a_ch[m] * (double)rhs_c[m - 1].im) * denom);
                            }
                            for (int m = n_tot - 2; m >= 0; m--) {
                                rhs_c[m].re = (tgk_real)((double)rhs_c[m].re - cp_ch[m] * (double)rhs_c[m + 1].re);
                                rhs_c[m].im = (tgk_real)((double)rhs_c[m].im - cp_ch[m] * (double)rhs_c[m + 1].im);
                            }

                            for (int l = 0; l < n_links; l++) {
                                int ikx = kx_ch[l];
                                for (int iz = 0; iz < nz; iz++) {
                                    h_slab[((size_t)iz * nky + iky) * nkx + ikx] = rhs_c[l * nz + iz];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    free(a_ch);
    free(b_ch);
    free(c_ch);
    free(cp_ch);
    free(z_ch);
    free(rhs_c);
    free(u_in);
    free(kx_ch);
    free(vis);
}
