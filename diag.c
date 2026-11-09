#include "tgk.h"
#include <stdlib.h>
#include <string.h>

#define TGK_SNAP_MAGIC  0x314B4754U /* "TGK1" */
#define TGK_CHECK_MAGIC 0x4B434754U /* "TGCK" */

void tgk_diag_compute(tgk_ctx *ctx, const tgk_complex *h_in, tgk_diag_out *out) {
    const tgk_grid  *gr = &ctx->grid;
    const tgk_geom  *gm = &ctx->geom;
    const tgk_param *p  = &ctx->param;

    const int nkx      = gr->nkx;
    const int nky      = gr->nky;
    const int nz       = gr->nz;
    const int nv       = gr->nv;
    const int nmu      = gr->nmu;
    const int nspecies = gr->nspecies;
    const size_t nplane = gr->nplane;
    const size_t nfield = gr->nfield;
    const int has_apar  = (p->beta > 0.0);

    memset(out, 0, sizeof(*out));

    double *z_phi2 = calloc(nz, sizeof(double));
    double *z_zf2  = calloc(nz, sizeof(double));
    double *ky_amp2 = calloc(nky, sizeof(double));
    tgk_complex64 *ky_cplx = calloc(nky, sizeof(tgk_complex64));

    /* 1. Field electrostatic & electromagnetic energy and ky spectrum */
    for (int iz = 0; iz < nz; iz++) {
        double wz = gm->z_weight[iz];
        size_t z_off = (size_t)iz * nplane;
        double p2_z = 0.0;
        double zf_z = 0.0;

        for (int iky = 0; iky < nky; iky++) {
            /* Reality factor: ky=0 modes count with weight 0.5 (since +kx and -kx are both stored),
             * while ky > 0 modes count with weight 1.0 (representing +ky and -ky combined). */
            double m_fac = (iky == 0) ? 0.5 : 1.0;
            for (int ikx = 0; ikx < nkx; ikx++) {
                size_t k = (size_t)iky * nkx + ikx;
                tgk_complex ph = ctx->phi[z_off + k];
                double a2 = tgk_cabs2(ph);
                p2_z += m_fac * a2;
                if (iky == 0) zf_z += m_fac * a2;
                ky_amp2[iky] += wz * m_fac * a2;
                if (ikx == 0 || iky > 0) {
                    ky_cplx[iky].re += wz * (double)ph.re;
                    ky_cplx[iky].im += wz * (double)ph.im;
                }
            }
        }
        z_phi2[iz] = wz * p2_z;
        z_zf2[iz]  = wz * zf_z;
    }

    out->phi2_total = tgk_pairwise_sum_f64(z_phi2, nz);
    out->phi2_zonal = tgk_pairwise_sum_f64(z_zf2, nz);

    /* 2. Instantaneous linear growth rate gamma and real frequency omega for dominant ky > 0 */
    double dt_diag = ctx->time - ctx->prev_time;
    int dom_iky = (nky > 1) ? 1 : 0;
    double max_ky_e = -1.0;
    for (int iky = 1; iky < nky; iky++) {
        if (ky_amp2[iky] > max_ky_e) {
            max_ky_e = ky_amp2[iky];
            dom_iky  = iky;
        }
    }

    for (int iky = 0; iky < nky; iky++) {
        double amp = sqrt(fmax(ky_amp2[iky], 1e-300));
        double phase = atan2(ky_cplx[iky].im, ky_cplx[iky].re);

        if (dt_diag > 1e-12 && ctx->prev_ky_amp[iky] > 1e-150 && amp > 1e-150) {
            double g = log(amp / ctx->prev_ky_amp[iky]) / dt_diag;
            double dphi = phase - ctx->prev_ky_phase[iky];
            while (dphi >  3.14159265358979323846) dphi -= 2.0 * 3.14159265358979323846;
            while (dphi < -3.14159265358979323846) dphi += 2.0 * 3.14159265358979323846;
            /* Convention: phi ~ exp(-i * omega * t + gamma * t) => omega = -d(phase)/dt */
            double w = -dphi / dt_diag;
            if (iky == dom_iky) {
                out->gamma_max = g;
                out->omega_max = w;
            }
        }
        ctx->prev_ky_amp[iky]   = amp;
        ctx->prev_ky_phase[iky] = phase;
    }

    /* 3. Heat flux Q_s, particle flux Gamma_s, and Free Energy W balance in FP64 */
    double *slab_fe = calloc((size_t)nspecies * nmu, sizeof(double));
    double *slab_q  = calloc((size_t)nspecies * nmu, sizeof(double));
    double *slab_g  = calloc((size_t)nspecies * nmu, sizeof(double));
    double *slab_co = calloc((size_t)nspecies * nmu, sizeof(double));
    double *slab_hd = calloc((size_t)nspecies * nmu, sizeof(double));
    const double hyp_z    = p->hyper_z / 16.0; /* must match rhs.c */

    for (int s = 0; s < nspecies; s++) {
        const tgk_species_param *sp = &p->spec[s];
        double ns   = sp->dens;
        double ts   = sp->temp;
        double vths = sqrt(ts / sp->mass);
        double z_t  = sp->charge / ts;
        /* IMEX electrons skip explicit streaming and its hyperdiffusion (see rhs.c) */
        const int has_hypz = (hyp_z > 0.0) && !(p->imex && sp->charge < 0.0);

        for (int imu = 0; imu < nmu; imu++) {
            double mu = gr->mu[imu];
            const tgk_real *j0_sm = gr->j0 + ((size_t)s * nmu + imu) * nfield;
            double fe_acc = 0.0, q_acc = 0.0, g_acc = 0.0, co_acc = 0.0, hd_acc = 0.0;

            for (int iv = 0; iv < nv; iv++) {
                double vpar  = gr->vpar[iv];
                tgk_real v_fac = (tgk_real)(vths * vpar);
                const tgk_complex *h_slab = h_in + (((size_t)s * nmu + imu) * nv + iv) * nfield;

                for (int iz = 0; iz < nz; iz++) {
                    double b  = gm->bmag[iz];
                    double wz = gm->z_weight[iz];
                    double energy = 0.5 * vpar * vpar + mu * b;
                    size_t vidx = (((size_t)s * nmu + imu) * nv + iv) * nz + iz;
                    double w_vel = gr->w_vel[vidx];
                    double f0    = (double)gr->f0[vidx];
                    double inv_f0 = (f0 > 1e-30) ? (1.0 / f0) : 0.0;

                    size_t z_off = (size_t)iz * nplane;
                    for (int iky = 0; iky < nky; iky++) {
                        double ky = gr->ky[iky];
                        double m_fac = (iky == 0) ? 0.5 : 1.0;

                        for (int ikx = 0; ikx < nkx; ikx++) {
                            size_t k = (size_t)iky * nkx + ikx;
                            tgk_complex hk  = h_slab[z_off + k];
                            tgk_complex ph  = ctx->phi[z_off + k];
                            double j0_val   = (double)j0_sm[z_off + k];

                            double h2 = tgk_cabs2(hk);
                            /* Entropy part of free energy: n_s T_s |h_s|^2 / (2 F_0s) */
                            double dfe = 0.5 * ns * ts * w_vel * inv_f0 * m_fac * wz * h2;
                            fe_acc += dfe;
                            co_acc -= 2.0 * sp->nu * dfe;

                            /* Exact power of the z-hyperdiffusion operator in the free-energy norm:
                             * sum (n T / F0) Re[ conj(h + Z/T F0 chi) * c_hypz * d4h ] */
                            if (has_hypz) {
                                tgk_complex hm2 = tgk_fetch_z_ts(gr, h_slab, iz - 2, iky, ikx);
                                tgk_complex hm1 = tgk_fetch_z_ts(gr, h_slab, iz - 1, iky, ikx);
                                tgk_complex hp1 = tgk_fetch_z_ts(gr, h_slab, iz + 1, iky, ikx);
                                tgk_complex hp2 = tgk_fetch_z_ts(gr, h_slab, iz + 2, iky, ikx);
                                double d4_re = (double)hm2.re - 4.0 * hm1.re + 6.0 * hk.re - 4.0 * hp1.re + hp2.re;
                                double d4_im = (double)hm2.im - 4.0 * hm1.im + 6.0 * hk.im - 4.0 * hp1.im + hp2.im;
                                double c_hypz = -hyp_z * fabs(vths * vpar * gm->gradpar[iz]) / gm->dz;
                                double cx_re = (double)ph.re, cx_im = (double)ph.im;
                                if (has_apar) {
                                    cx_re -= (double)v_fac * ctx->apar[z_off + k].re;
                                    cx_im -= (double)v_fac * ctx->apar[z_off + k].im;
                                }
                                double H_re = hk.re + z_t * f0 * j0_val * cx_re;
                                double H_im = hk.im + z_t * f0 * j0_val * cx_im;
                                hd_acc += ns * ts * w_vel * inv_f0 * m_fac * wz * c_hypz *
                                          (H_re * d4_re + H_im * d4_im);
                            }

                            if (iky > 0) {
                                /* Radial ExB velocity v_ExB_x = -dchi/dy = -i * ky * chi
                                 * Flux = Re[ (-i * ky * chi)^* * h_s ] = ky * (chi.im * h.re - chi.re * h.im) */
                                tgk_complex chi = ph;
                                if (has_apar) {
                                    chi.re -= v_fac * ctx->apar[z_off + k].re;
                                    chi.im -= v_fac * ctx->apar[z_off + k].im;
                                }
                                double chi_re = j0_val * (double)chi.re;
                                double chi_im = j0_val * (double)chi.im;
                                double v_exb_h = ky * (chi_im * (double)hk.re - chi_re * (double)hk.im);

                                g_acc += wz * m_fac * ns * w_vel * v_exb_h;
                                q_acc += wz * m_fac * ns * ts * w_vel * energy * v_exb_h;
                            }
                        }
                    }
                }
            }
            size_t sm_idx = (size_t)s * nmu + imu;
            slab_fe[sm_idx] = fe_acc;
            slab_q[sm_idx]  = q_acc;
            slab_g[sm_idx]  = g_acc;
            slab_co[sm_idx] = co_acc;
            slab_hd[sm_idx] = hd_acc;
        }
    }

    /* Field polarization + magnetic part of free energy:
     * - sum_s (Z_s^2 n_s / 2 T_s) |phi|^2 + (1/2) sum_k D_pol |phi|^2 + (1 / 2*beta) k_perp^2 |A_par|^2 */
    double fe_field = 0.0;
    for (int iz = 0; iz < nz; iz++) {
        double wz = gm->z_weight[iz];
        size_t z_off = (size_t)iz * nplane;
        for (int iky = 0; iky < nky; iky++) {
            double m_fac = (iky == 0) ? 0.5 : 1.0;
            for (int ikx = 0; ikx < nkx; ikx++) {
                size_t k = z_off + (size_t)iky * nkx + ikx;
                double phi2 = tgk_cabs2(ctx->phi[k]);
                fe_field += 0.5 * wz * m_fac * gr->qn_denom[k] * phi2;
                if (has_apar) {
                    double apar2 = tgk_cabs2(ctx->apar[k]);
                    fe_field += 0.5 * wz * m_fac * (gr->kperp2[k] / p->beta) * apar2;
                }
            }
        }
    }

    for (int s = 0; s < nspecies; s++) {
        double q_s = tgk_pairwise_sum_f64(slab_q + (size_t)s * nmu, nmu);
        double g_s = tgk_pairwise_sum_f64(slab_g + (size_t)s * nmu, nmu);
        if (p->spec[s].charge > 0.0) {
            out->heat_flux_i += q_s;
            out->part_flux_i += g_s;
            double grad_ti = fmax(p->spec[s].tprim / p->rmaj, 1e-12);
            out->chi_i = out->heat_flux_i / (p->spec[s].dens * p->spec[s].temp * grad_ti);
        } else {
            out->heat_flux_e += q_s;
            out->part_flux_e += g_s;
            double grad_te = fmax(p->spec[s].tprim / p->rmaj, 1e-12);
            out->chi_e = out->heat_flux_e / (p->spec[s].dens * p->spec[s].temp * grad_te);
        }
        out->fe_drive += (p->spec[s].fprim * g_s * p->spec[s].temp +
                          p->spec[s].tprim * (q_s - 1.5 * g_s * p->spec[s].temp)) / p->rmaj;
    }

    out->free_energy = tgk_pairwise_sum_f64(slab_fe, (size_t)nspecies * nmu) + fe_field;
    out->fe_coll     = tgk_pairwise_sum_f64(slab_co, (size_t)nspecies * nmu);
    out->fe_dissip   = tgk_pairwise_sum_f64(slab_hd, (size_t)nspecies * nmu);

    if (dt_diag > 1e-12) {
        double dfe_dt = (out->free_energy - ctx->prev_free_energy) / dt_diag;
        out->fe_residual = dfe_dt - (out->fe_drive + out->fe_dissip + out->fe_coll);
    }

    ctx->prev_time        = ctx->time;
    ctx->prev_free_energy = out->free_energy;

    free(z_phi2);
    free(z_zf2);
    free(ky_amp2);
    free(ky_cplx);
    free(slab_fe);
    free(slab_q);
    free(slab_g);
    free(slab_co);
    free(slab_hd);
}

void tgk_diag_write_trace(tgk_ctx *ctx, FILE *fp, const tgk_diag_out *d) {
    if (!fp) return;
    if (ctx->step == 0) {
        fprintf(fp, "# step time dt phi2_tot phi2_zf Q_i Q_e chi_i gamma omega W_free W_res\n");
    }
    fprintf(fp, "%6d %12.6e %12.6e %14.6e %14.6e %14.6e %14.6e %14.6e %13.6e %13.6e %14.6e %13.6e\n",
            ctx->step, ctx->time, ctx->dt,
            d->phi2_total, d->phi2_zonal,
            d->heat_flux_i, d->heat_flux_e, d->chi_i,
            d->gamma_max, d->omega_max,
            d->free_energy, d->fe_residual);
    fflush(fp);
}

/* Fixed 64-byte binary header for field and moment snapshots:
 *   uint32_t magic      = 0x314B4754 ("TGK1")
 *   uint32_t version    = 1
 *   uint32_t precision  = sizeof(tgk_real) (4 or 8)
 *   int32_t  step
 *   double   time
 *   int32_t  nkx, nky, nz, nspecies
 *   uint8_t  reserved[20] */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t precision;
    int32_t  step;
    double   time;
    int32_t  nkx;
    int32_t  nky;
    int32_t  nz;
    int32_t  nspecies;
    uint8_t  reserved[20];
} tgk_snap_header;

int tgk_diag_write_snapshot(const tgk_ctx *ctx, const char *filename) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) return -1;

    tgk_snap_header hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic     = TGK_SNAP_MAGIC;
    hdr.version   = 1;
    hdr.precision = (uint32_t)sizeof(tgk_real);
    hdr.step      = ctx->step;
    hdr.time      = ctx->time;
    hdr.nkx       = ctx->grid.nkx;
    hdr.nky       = ctx->grid.nky;
    hdr.nz        = ctx->grid.nz;
    hdr.nspecies  = ctx->grid.nspecies;

    fwrite(&hdr, sizeof(hdr), 1, fp);
    fwrite(ctx->phi,  sizeof(tgk_complex), ctx->grid.nfield, fp);
    fwrite(ctx->apar, sizeof(tgk_complex), ctx->grid.nfield, fp);
    fclose(fp);
    return 0;
}

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t precision;
    int32_t  nthreads;
    int32_t  step;
    double   time;
    double   dt;
    int32_t  nkx, nky, nz, nv, nmu, nspecies;
} tgk_check_header;

int tgk_checkpoint_write(const tgk_ctx *ctx, const char *filename) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) return -1;

    tgk_check_header hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic     = TGK_CHECK_MAGIC;
    hdr.version   = 1;
    hdr.precision = (uint32_t)sizeof(tgk_real);
    hdr.nthreads  = ctx->nthreads;
    hdr.step      = ctx->step;
    hdr.time      = ctx->time;
    hdr.dt        = ctx->dt;
    hdr.nkx       = ctx->grid.nkx;
    hdr.nky       = ctx->grid.nky;
    hdr.nz        = ctx->grid.nz;
    hdr.nv        = ctx->grid.nv;
    hdr.nmu       = ctx->grid.nmu;
    hdr.nspecies  = ctx->grid.nspecies;

    fwrite(&hdr, sizeof(hdr), 1, fp);
    fwrite(ctx->h,    sizeof(tgk_complex), ctx->grid.ntot,   fp);
    fwrite(ctx->phi,  sizeof(tgk_complex), ctx->grid.nfield, fp);
    fwrite(ctx->apar, sizeof(tgk_complex), ctx->grid.nfield, fp);
    fclose(fp);
    return 0;
}

int tgk_checkpoint_read(tgk_ctx *ctx, const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) return -1;

    tgk_check_header hdr;
    if (fread(&hdr, sizeof(hdr), 1, fp) != 1) { fclose(fp); return -1; }
    if (hdr.magic != TGK_CHECK_MAGIC || hdr.precision != sizeof(tgk_real)) { fclose(fp); return -1; }
    if (hdr.nkx != ctx->grid.nkx || hdr.nky != ctx->grid.nky || hdr.nz != ctx->grid.nz ||
        hdr.nv  != ctx->grid.nv  || hdr.nmu != ctx->grid.nmu || hdr.nspecies != ctx->grid.nspecies) {
        fclose(fp);
        return -1;
    }

    ctx->step = hdr.step;
    ctx->time = hdr.time;
    ctx->dt   = hdr.dt;

    if (fread(ctx->h,    sizeof(tgk_complex), ctx->grid.ntot,   fp) != ctx->grid.ntot)   { fclose(fp); return -1; }
    if (fread(ctx->phi,  sizeof(tgk_complex), ctx->grid.nfield, fp) != ctx->grid.nfield) { fclose(fp); return -1; }
    if (fread(ctx->apar, sizeof(tgk_complex), ctx->grid.nfield, fp) != ctx->grid.nfield) { fclose(fp); return -1; }
    fclose(fp);
    return 0;
}
