#include "tgk.h"
#include <string.h>

void tgk_field_clear_accum(tgk_ctx *ctx) {
    size_t nfield = ctx->grid.nfield;
    for (int t = 0; t < ctx->nthreads; t++) {
        memset(ctx->ws[t].qn_num_loc, 0, sizeof(tgk_complex64) * nfield);
        if (ctx->param.beta > 0.0) {
            memset(ctx->ws[t].amp_num_loc, 0, sizeof(tgk_complex64) * nfield);
        }
        ctx->ws[t].vmax_nl = 0.0;
        ctx->ws[t].cached_sm = -1;
    }
}

/* Reduce thread-local FP64 velocity-space integrals using pairwise summation across threads
 * and solve the algebraic Quasineutrality and parallel Ampere equations at every (kx, ky, z). */
void tgk_field_reduce_and_solve(tgk_ctx *ctx) {
    const tgk_grid *gr  = &ctx->grid;
    const tgk_geom *gm  = &ctx->geom;
    const tgk_param *p  = &ctx->param;
    const int nkx       = gr->nkx;
    const int nky       = gr->nky;
    const int nz        = gr->nz;
    const size_t nfield = gr->nfield;
    const int nt        = ctx->nthreads;
    const int has_apar  = (p->beta > 0.0);

    /* 1. Reduce thread-local accumulators into ctx->qn_num and ctx->amp_num */
    for (size_t i = 0; i < nfield; i++) {
        double re_q = 0.0, im_q = 0.0;
        for (int t = 0; t < nt; t++) {
            re_q += ctx->ws[t].qn_num_loc[i].re;
            im_q += ctx->ws[t].qn_num_loc[i].im;
        }
        ctx->qn_num[i].re = re_q;
        ctx->qn_num[i].im = im_q;

        if (has_apar) {
            double re_a = 0.0, im_a = 0.0;
            for (int t = 0; t < nt; t++) {
                re_a += ctx->ws[t].amp_num_loc[i].re;
                im_a += ctx->ws[t].amp_num_loc[i].im;
            }
            ctx->amp_num[i].re = re_a;
            ctx->amp_num[i].im = im_a;
        }
    }

    /* 2. Solve for phi and apar at non-zonal modes (ky > 0) and kinetic-electron zonal modes */
    for (int iz = 0; iz < nz; iz++) {
        for (int iky = 0; iky < nky; iky++) {
            if (iky == 0 && p->adiabatic_electrons) continue; /* handled below with FS average */

            for (int ikx = 0; ikx < nkx; ikx++) {
                size_t idx = ((size_t)iz * nky + iky) * nkx + ikx;
                if (iky == 0 && ikx == 0) {
                    ctx->phi[idx]  = (tgk_complex){0.0f, 0.0f};
                    ctx->apar[idx] = (tgk_complex){0.0f, 0.0f};
                    continue;
                }

                double denom_q = gr->qn_denom[idx];
                if (denom_q > 1e-18) {
                    ctx->phi[idx].re = (tgk_real)(ctx->qn_num[idx].re / denom_q);
                    ctx->phi[idx].im = (tgk_real)(ctx->qn_num[idx].im / denom_q);
                } else {
                    ctx->phi[idx] = (tgk_complex){0.0f, 0.0f};
                }

                if (has_apar) {
                    double denom_a = gr->amp_denom[idx];
                    if (denom_a > 1e-18) {
                        ctx->apar[idx].re = (tgk_real)(ctx->amp_num[idx].re / denom_a);
                        ctx->apar[idx].im = (tgk_real)(ctx->amp_num[idx].im / denom_a);
                    } else {
                        ctx->apar[idx] = (tgk_complex){0.0f, 0.0f};
                    }
                } else {
                    ctx->apar[idx] = (tgk_complex){0.0f, 0.0f};
                }
            }
        }
    }

    /* 3. Adiabatic-electron Zonal Flow (ky = 0) solve with flux-surface-averaged correction:
     * [Q_pol(kx, z) + tau_e] * phi(kx, z) - tau_e * <phi(kx, z)>_FS = N(kx, z)
     * => phi(kx, z) = (N(kx, z) + tau_e * <phi(kx)>_FS) / (Q_pol(kx, z) + tau_e) */
    if (p->adiabatic_electrons) {
        for (int ikx = 0; ikx < nkx; ikx++) {
            if (ikx == 0) {
                for (int iz = 0; iz < nz; iz++) {
                    size_t idx = ((size_t)iz * nky + 0) * nkx + 0;
                    ctx->phi[idx]  = (tgk_complex){0.0f, 0.0f};
                    ctx->apar[idx] = (tgk_complex){0.0f, 0.0f};
                }
                continue;
            }

            /* Compute <N / D_tot>_FS across z in FP64 */
            double avg_re = 0.0, avg_im = 0.0;
            for (int iz = 0; iz < nz; iz++) {
                size_t idx = ((size_t)iz * nky + 0) * nkx + ikx;
                double d_tot = gr->qn_denom[idx] + p->tau_e;
                double w_dt  = gm->z_weight[iz] / d_tot;
                avg_re += w_dt * ctx->qn_num[idx].re;
                avg_im += w_dt * ctx->qn_num[idx].im;
            }

            /* tau_e * <phi>_FS = zf_fac[ikx] * <N / D_tot>_FS */
            double zf_corr_re = gr->zf_fac[ikx] * avg_re;
            double zf_corr_im = gr->zf_fac[ikx] * avg_im;

            for (int iz = 0; iz < nz; iz++) {
                size_t idx = ((size_t)iz * nky + 0) * nkx + ikx;
                double d_tot = gr->qn_denom[idx] + p->tau_e;
                ctx->phi[idx].re = (tgk_real)((ctx->qn_num[idx].re + zf_corr_re) / d_tot);
                ctx->phi[idx].im = (tgk_real)((ctx->qn_num[idx].im + zf_corr_im) / d_tot);
                ctx->apar[idx]   = (tgk_complex){0.0f, 0.0f};
            }
        }
    }
}

/* Standalone field computation from an arbitrary distribution function h_in
 * (used once at initialization or restart before the fused RK loop begins). */
void tgk_field_compute_from_h(tgk_ctx *ctx, const tgk_complex *h_in) {
    const tgk_grid *gr  = &ctx->grid;
    const tgk_param *p  = &ctx->param;
    const int nkx       = gr->nkx;
    const int nky       = gr->nky;
    const int nz        = gr->nz;
    const int nv        = gr->nv;
    const int nmu       = gr->nmu;
    const int nspecies  = gr->nspecies;
    const size_t nfield = gr->nfield;
    const int has_apar  = (p->beta > 0.0);

    tgk_field_clear_accum(ctx);
    tgk_complex64 *qn_loc  = ctx->ws[0].qn_num_loc;
    tgk_complex64 *amp_loc = ctx->ws[0].amp_num_loc;

    for (int s = 0; s < nspecies; s++) {
        double zs   = p->spec[s].charge;
        double ns   = p->spec[s].dens;
        double vths = sqrt(p->spec[s].temp / p->spec[s].mass);
        double q_fac = zs * ns;
        double a_fac = 0.5 * p->beta * zs * ns * vths;

        for (int imu = 0; imu < nmu; imu++) {
            const tgk_real *j0_sm = gr->j0 + ((size_t)s * nmu + imu) * nfield;

            for (int iv = 0; iv < nv; iv++) {
                double vpar = gr->vpar[iv];
                const tgk_complex *h_slab = h_in + (((size_t)s * nmu + imu) * nv + iv) * nfield;

                for (int iz = 0; iz < nz; iz++) {
                    size_t vidx = (((size_t)s * nmu + imu) * nv + iv) * nz + iz;
                    double w_v  = gr->w_vel[vidx];
                    double wq   = q_fac * w_v;
                    double wa   = a_fac * vpar * w_v;

                    size_t z_off = (size_t)iz * nky * nkx;
                    const tgk_complex *h_z  = h_slab + z_off;
                    const tgk_real    *j0_z = j0_sm  + z_off;
                    tgk_complex64     *qn_z = qn_loc + z_off;
                    tgk_complex64     *am_z = amp_loc + z_off;

                    size_t nplane = (size_t)nky * nkx;
                    for (size_t k = 0; k < nplane; k++) {
                        double j0_val = (double)j0_z[k];
                        double h_re   = (double)h_z[k].re;
                        double h_im   = (double)h_z[k].im;
                        qn_z[k].re += wq * j0_val * h_re;
                        qn_z[k].im += wq * j0_val * h_im;
                        if (has_apar) {
                            am_z[k].re += wa * j0_val * h_re;
                            am_z[k].im += wa * j0_val * h_im;
                        }
                    }
                }
            }
        }
    }

    tgk_field_reduce_and_solve(ctx);
}
