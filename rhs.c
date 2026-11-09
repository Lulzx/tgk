#include "tgk.h"
#include <stdatomic.h>
#include <string.h>

/* Fetch a plane element at (iz_target, iky, ikx) applying twist-and-shift boundary condition
 * when iz_target crosses the ends of the flux tube [0, nz-1]. */
static inline tgk_complex fetch_z_ts(const tgk_grid *gr, const tgk_complex *slab_base,
                                     int iz_target, int iky, int ikx) {
    const int nz  = gr->nz;
    const int nkx = gr->nkx;
    const int nky = gr->nky;

    if ((unsigned)iz_target < (unsigned)nz) {
        return slab_base[((size_t)iz_target * nky + iky) * nkx + ikx];
    } else if (iz_target < 0) {
        int ikx_conn = gr->ts_minus[(size_t)iky * nkx + ikx];
        if (ikx_conn < 0) return (tgk_complex){0.0f, 0.0f};
        return slab_base[((size_t)(iz_target + nz) * nky + iky) * nkx + ikx_conn];
    } else {
        int ikx_conn = gr->ts_plus[(size_t)iky * nkx + ikx];
        if (ikx_conn < 0) return (tgk_complex){0.0f, 0.0f};
        return slab_base[((size_t)(iz_target - nz) * nky + iky) * nkx + ikx_conn];
    }
}

/* Fetch gyroaveraged potential chi = J0 * (phi - v_th * vpar * apar) at (iz_target, iky, ikx)
 * with twist-and-shift boundary conditions. */
static inline tgk_complex fetch_chi_ts(const tgk_grid *gr, const tgk_complex *phi,
                                       const tgk_complex *apar, const tgk_real *j0_sm,
                                       tgk_real v_fac, int has_apar,
                                       int iz_target, int iky, int ikx) {
    const int nz  = gr->nz;
    const int nkx = gr->nkx;
    const int nky = gr->nky;
    int iz_eff = iz_target;
    int ikx_eff = ikx;

    if (iz_target < 0) {
        ikx_eff = gr->ts_minus[(size_t)iky * nkx + ikx];
        if (ikx_eff < 0) return (tgk_complex){0.0f, 0.0f};
        iz_eff = iz_target + nz;
    } else if (iz_target >= nz) {
        ikx_eff = gr->ts_plus[(size_t)iky * nkx + ikx];
        if (ikx_eff < 0) return (tgk_complex){0.0f, 0.0f};
        iz_eff = iz_target - nz;
    }

    size_t idx = ((size_t)iz_eff * nky + iky) * nkx + ikx_eff;
    tgk_real j0 = j0_sm[idx];
    tgk_complex p = phi[idx];
    if (has_apar) {
        tgk_complex a = apar[idx];
        p.re -= v_fac * a.re;
        p.im -= v_fac * a.im;
    }
    return tgk_cscale(j0, p);
}

/* Evaluate one fused RK stage across all (species, mu, v_parallel) slabs via work-stealing.
 *
 * Memory budget per stage:
 * - Reads:  h_in (current stage distribution function), h (base state on stage > 0)
 * - Writes: h_rhs (accumulated RK update), h_stage (next stage state), plus FP64 field integral
 *
 * Physics terms included in single slab pass:
 * 1. Parallel streaming: -v_th_s * v_par * b_hat.grad(z) * d(h + Z_s/T_s F0 J0 phi)/dz
 *    + fourth-order parallel hyperdiffusion -hyper_z * d^4 h / dz^4
 * 2. Mirror force: +v_th_s * mu * dB/dz * b_hat.grad(z) * dh/dv_par
 * 3. Magnetic curvature & grad-B drifts:
 *    -i * omega_d(kx, ky, z, v_par, mu) * (h + Z_s/T_s F0 chi)
 * 4. Equilibrium density & temperature gradient drive:
 *    -i * ky * (a/L_n + (v_par^2/2 + mu*B - 3/2) * a/L_T) * F0 * chi,  a/L = (R/L) / (R/a)
 * 5. Nonlinear E x B advection:
 *    - { chi, h } = -(dchi/dx * dh/dy - dchi/dy * dh/dx) via pseudo-spectral 2/3-dealiased FFTs. */
void tgk_rhs_eval_fused_stage(tgk_ctx *ctx, int tid) {
    const tgk_grid  *gr = &ctx->grid;
    const tgk_geom  *gm = &ctx->geom;
    const tgk_param *p  = &ctx->param;
    tgk_thread_ws   *ws = &ctx->ws[tid];

    const int nkx      = gr->nkx;
    const int nky      = gr->nky;
    const int nz       = gr->nz;
    const int nv       = gr->nv;
    const int nmu      = gr->nmu;
    const size_t nplane = gr->nplane;
    const size_t nfield = gr->nfield;

    const int stage_idx   = ctx->rk_stage_idx;
    const tgk_real dt_b   = (tgk_real)(ctx->rk_dt * ctx->rk_b);
    const tgk_real dt_c   = (tgk_real)(ctx->rk_dt * ctx->rk_c);
    const int is_last_stg = (stage_idx == 3);
    const int has_apar    = (p->beta > 0.0);
    const int do_nl       = p->nonlinear;

    /* Input distribution for this stage: stage 0 reads ctx->h, stages 1..3 read ctx->h_stage */
    const tgk_complex *h_in = (stage_idx == 0) ? ctx->h : ctx->h_stage;

    const double inv_12dz = 1.0 / (12.0 * gm->dz);
    const double inv_12dv = 1.0 / (12.0 * gr->dv);
    const double hyp_z    = p->hyper_z / 16.0; /* normalized grid-scale 4th-order damping */
    const int nx_pad      = ctx->fft.nx_pad;
    const int ny_pad      = ctx->fft.ny_pad;
    const size_t n_real   = (size_t)nx_pad * (size_t)ny_pad;

    /* Scratch pointers in thread workspace */
    tgk_complex *spec_tmp1 = ws->cx_buf1;
    tgk_complex *spec_tmp2 = ws->cx_buf1 + nplane;
    tgk_complex *fft_scr1  = ws->cx_buf1 + 2 * nplane;
    tgk_complex *fft_scr2  = ws->cx_buf2;

    double vmax_nl_local = ws->vmax_nl;

    /* Work-stealing loop: thread tid starts with its own contiguous slab range in L2 cache,
     * and steals from peer queues if it finishes early. */
    const int nt = ctx->nthreads;
    for (int t_offset = 0; t_offset < 1; t_offset++) {
        int victim = (tid + t_offset) % nt;
        while (1) {
            int slab_id = atomic_fetch_add_explicit(&ctx->steal_q[victim], 1, memory_order_relaxed);
            if (slab_id >= ctx->steal_end[victim]) break;

        int iv  = slab_id % nv;
        int tmp = slab_id / nv;
        int imu = tmp % nmu;
        int s   = tmp / nmu;

        const tgk_species_param *sp = &p->spec[s];
        const double zs   = sp->charge;
        const double ts   = sp->temp;
        const double ms   = sp->mass;
        const double ns   = sp->dens;
        const double vths = sqrt(ts / ms);
        const double z_t  = zs / ts;
        const double vpar = gr->vpar[iv];
        const double mu   = gr->mu[imu];
        const tgk_real v_fac = (tgk_real)vths * (tgk_real)vpar;

        /* Check if implicit electron streaming handles parallel streaming for this species */
        const int skip_par_stream = (p->imex && zs < 0.0);

        const size_t slab_offset = (size_t)slab_id * nfield;
        const tgk_complex *h_slab = h_in + slab_offset;
        const tgk_real    *j0_sm  = gr->j0 + ((size_t)s * nmu + imu) * nfield;

        /* Pointers to adjacent v_parallel slabs for 4th-order mirror force term */
        const tgk_complex *h_vm2 = (iv >= 2)     ? (h_slab - 2 * nfield) : NULL;
        const tgk_complex *h_vm1 = (iv >= 1)     ? (h_slab - 1 * nfield) : NULL;
        const tgk_complex *h_vp1 = (iv + 1 < nv) ? (h_slab + 1 * nfield) : NULL;
        const tgk_complex *h_vp2 = (iv + 2 < nv) ? (h_slab + 2 * nfield) : NULL;

        const double q_fac = zs * ns;
        const double a_fac = 0.5 * p->beta * zs * ns * vths * vpar;

        /* Precompute gyroaveraged field derivatives in real space once per (s, mu) across all z,
         * reusing them across all nv = 32 v_parallel slabs (saves 2 * nz 2D FFTs per slab!) */
        const int sm_id = s * nmu + imu;
        if (do_nl && ws->cached_sm != sm_id) {
            ws->cached_sm = sm_id;
            for (int iz = 0; iz < nz; iz++) {
                const size_t z_off      = (size_t)iz * nplane;
                const size_t r_off      = (size_t)iz * n_real;
                const tgk_complex *phi_z  = ctx->phi  + z_off;
                const tgk_complex *apar_z = ctx->apar + z_off;
                const tgk_real    *j0_z   = j0_sm     + z_off;

                for (int iky = 0; iky < nky; iky++) {
                    tgk_real ky = (tgk_real)gr->ky[iky];
                    size_t row = (size_t)iky * nkx;
                    for (int ikx = 0; ikx < nkx; ikx++) {
                        tgk_real kx = (tgk_real)gr->kx[ikx];
                        size_t k = row + ikx;
                        tgk_complex jp = tgk_cscale(j0_z[k], phi_z[k]);
                        spec_tmp1[k] = tgk_cmul_is(kx, jp);
                        spec_tmp2[k] = tgk_cmul_is(ky, jp);
                    }
                }
                tgk_fft_c2r_pair_dealiased(&ctx->fft, spec_tmp1, spec_tmp2,
                                           ws->rx_phi_dx_z + r_off, ws->rx_phi_dy_z + r_off,
                                           fft_scr1, fft_scr2);

                if (has_apar) {
                    for (int iky = 0; iky < nky; iky++) {
                        tgk_real ky = (tgk_real)gr->ky[iky];
                        size_t row = (size_t)iky * nkx;
                        for (int ikx = 0; ikx < nkx; ikx++) {
                            tgk_real kx = (tgk_real)gr->kx[ikx];
                            size_t k = row + ikx;
                            tgk_complex ja = tgk_cscale(j0_z[k], apar_z[k]);
                            spec_tmp1[k] = tgk_cmul_is(kx, ja);
                            spec_tmp2[k] = tgk_cmul_is(ky, ja);
                        }
                    }
                    tgk_fft_c2r_pair_dealiased(&ctx->fft, spec_tmp1, spec_tmp2,
                                               ws->rx_apar_dx_z + r_off, ws->rx_apar_dy_z + r_off,
                                               fft_scr1, fft_scr2);
                }
            }
        }

        for (int iz = 0; iz < nz; iz++) {
            const size_t z_off = (size_t)iz * nplane;
            const double b     = gm->bmag[iz];
            const double gpar  = gm->gradpar[iz];
            const double dbdz  = gm->dbdz[iz];
            const double cv_x  = gm->cv_x[iz];
            const double cv_y  = gm->cv_y[iz];

            const size_t vidx  = (((size_t)s * nmu + imu) * nv + iv) * nz + iz;
            const tgk_real f0_val = gr->f0[vidx];
            const double w_vel    = gr->w_vel[vidx];
            const double wq       = q_fac * w_vel;
            const double wa       = a_fac * w_vel;

            /* 1. If nonlinear, compute pseudo-spectral Poisson bracket -{chi, h} on this z-plane */
            if (do_nl) {
                const size_t r_off      = (size_t)iz * n_real;
                const tgk_real *pdx     = ws->rx_phi_dx_z  + r_off;
                const tgk_real *pdy     = ws->rx_phi_dy_z  + r_off;
                const tgk_real *adx     = ws->rx_apar_dx_z + r_off;
                const tgk_real *ady     = ws->rx_apar_dy_z + r_off;
                const tgk_complex *hz   = h_slab           + z_off;

                tgk_fft_c2r_grad_dealiased(&ctx->fft, hz, gr->kx, gr->ky,
                                           ws->rx_dh_dx, ws->rx_dh_dy, fft_scr1, fft_scr2);

                /* Pointwise Poisson bracket in real space:
                 * nl_rhs = -(dchi/dx * dh/dy - dchi/dy * dh/dx) = dchi/dy * dh/dx - dchi/dx * dh/dy */
                tgk_real max_vx = 0.0f, max_vy = 0.0f;
#if TGK_HAS_NEON && !TGK_REAL_IS_FP64
                float32x4_t vmax_x_v = vdupq_n_f32(0.0f);
                float32x4_t vmax_y_v = vdupq_n_f32(0.0f);
                size_t r = 0;
                for (; r + 4 <= n_real; r += 4) {
                    float32x4_t dcx = vld1q_f32(pdx + r);
                    float32x4_t dcy = vld1q_f32(pdy + r);
                    if (has_apar) {
                        dcx = vfmsq_n_f32(dcx, vld1q_f32(adx + r), v_fac);
                        dcy = vfmsq_n_f32(dcy, vld1q_f32(ady + r), v_fac);
                    }
                    float32x4_t dhx = vld1q_f32(ws->rx_dh_dx + r);
                    float32x4_t dhy = vld1q_f32(ws->rx_dh_dy + r);
                    float32x4_t brk = vfmsq_f32(vmulq_f32(dcy, dhx), dcx, dhy);
                    vst1q_f32(ws->rx_nl + r, brk);
                    vmax_x_v = vmaxq_f32(vmax_x_v, vabsq_f32(dcy));
                    vmax_y_v = vmaxq_f32(vmax_y_v, vabsq_f32(dcx));
                }
                max_vx = vmaxvq_f32(vmax_x_v);
                max_vy = vmaxvq_f32(vmax_y_v);
                for (; r < n_real; r++) {
                    tgk_real dcx = pdx[r] - (has_apar ? v_fac * adx[r] : 0.0f);
                    tgk_real dcy = pdy[r] - (has_apar ? v_fac * ady[r] : 0.0f);
                    ws->rx_nl[r] = dcy * ws->rx_dh_dx[r] - dcx * ws->rx_dh_dy[r];
                    if (fabsf(dcy) > max_vx) max_vx = fabsf(dcy);
                    if (fabsf(dcx) > max_vy) max_vy = fabsf(dcx);
                }
#else
                for (size_t r = 0; r < n_real; r++) {
                    tgk_real dcx = pdx[r] - (has_apar ? v_fac * adx[r] : 0.0f);
                    tgk_real dcy = pdy[r] - (has_apar ? v_fac * ady[r] : 0.0f);
                    ws->rx_nl[r] = dcy * ws->rx_dh_dx[r] - dcx * ws->rx_dh_dy[r];
                    if (fabs(dcy) > max_vx) max_vx = fabs(dcy);
                    if (fabs(dcx) > max_vy) max_vy = fabs(dcx);
                }
#endif
                if (imu == 0 && iv == nv / 2) {
                    double kx_max = fabs(gr->kx[nkx / 2]);
                    double ky_max = gr->ky[nky - 1];
                    double rate = (double)max_vx * kx_max + (double)max_vy * ky_max;
                    if (rate > vmax_nl_local) vmax_nl_local = rate;
                }

                /* Transform nonlinear bracket back to dealiased spectral space spec_tmp1 */
                tgk_fft_r2c_dealiased(&ctx->fft, ws->rx_nl, spec_tmp1, fft_scr1, fft_scr2);
            }

            /* Precompute z-plane scalar coefficients */
            const tgk_real c_stream = skip_par_stream ? 0.0f : (tgk_real)(-vths * vpar * gpar * inv_12dz);
            const tgk_real c_mirror = (tgk_real)(vths * mu * dbdz * gpar * inv_12dv);
            const tgk_real c_hypz   = (tgk_real)(-hyp_z * fabs(vths * vpar * gpar) / gm->dz);
            const tgk_real c_zt_f0  = (tgk_real)(z_t) * f0_val;

            /* Energy factor for temperature-gradient drive: E/T_s - 3/2 = v_par^2/2 + mu*B - 1.5 */
            const double e_norm     = 0.5 * vpar * vpar + mu * b - 1.5;
            /* fprim, tprim are R/L_n, R/L_T; lengths are normalized to a, so divide by R/a */
            const double omega_star_fac = -(sp->fprim + e_norm * sp->tprim) / p->rmaj * (double)f0_val;
            /* Magnetic drift velocity factor: (T_s / Z_s) * (v_par^2 + mu*B) */
            const double v_drift    = (ts / zs) * (vpar * vpar + mu * b);

            const tgk_complex *phi_z  = ctx->phi  + z_off;
            const tgk_complex *apar_z = ctx->apar + z_off;
            const tgk_real    *j0_z   = j0_sm     + z_off;
            tgk_complex64     *qn_z   = ws->qn_num_loc  + z_off;
            tgk_complex64     *am_z   = ws->amp_num_loc + z_off;
#if TGK_HAS_NEON && !TGK_REAL_IS_FP64
            const int interior_z      = (iz >= 2 && iz < nz - 2);
            const int has_mirror      = (fabs(c_mirror) > 1e-18f);
#endif

            for (int iky = 0; iky < nky; iky++) {
                const double ky = gr->ky[iky];
                const tgk_real w_star = (tgk_real)(ky * omega_star_fac);
#if TGK_HAS_NEON && !TGK_REAL_IS_FP64
                const tgk_real wd_y   = (tgk_real)(v_drift * cv_y * ky);
                const tgk_real wd_x   = (tgk_real)(v_drift * cv_x);
#endif

                int ikx = 0;
#if TGK_HAS_NEON && !TGK_REAL_IS_FP64
                if (interior_z) {
                    const tgk_complex *hz_m2 = h_slab + z_off - 2 * nplane + (size_t)iky * nkx;
                    const tgk_complex *hz_m1 = h_slab + z_off - 1 * nplane + (size_t)iky * nkx;
                    const tgk_complex *hz_0  = h_slab + z_off              + (size_t)iky * nkx;
                    const tgk_complex *hz_p1 = h_slab + z_off + 1 * nplane + (size_t)iky * nkx;
                    const tgk_complex *hz_p2 = h_slab + z_off + 2 * nplane + (size_t)iky * nkx;

                    const tgk_complex *pz_m2 = ctx->phi + z_off - 2 * nplane + (size_t)iky * nkx;
                    const tgk_complex *pz_m1 = ctx->phi + z_off - 1 * nplane + (size_t)iky * nkx;
                    const tgk_complex *pz_0  = ctx->phi + z_off              + (size_t)iky * nkx;
                    const tgk_complex *pz_p1 = ctx->phi + z_off + 1 * nplane + (size_t)iky * nkx;
                    const tgk_complex *pz_p2 = ctx->phi + z_off + 2 * nplane + (size_t)iky * nkx;

                    const tgk_real *jz_m2 = j0_sm + z_off - 2 * nplane + (size_t)iky * nkx;
                    const tgk_real *jz_m1 = j0_sm + z_off - 1 * nplane + (size_t)iky * nkx;
                    const tgk_real *jz_0  = j0_sm + z_off              + (size_t)iky * nkx;
                    const tgk_real *jz_p1 = j0_sm + z_off + 1 * nplane + (size_t)iky * nkx;
                    const tgk_real *jz_p2 = j0_sm + z_off + 2 * nplane + (size_t)iky * nkx;

                    /* Skip (iky=0, ikx=0..1) in NEON loop only on odd nkx; actually just handle (0,0) after! */
                    for (; ikx + 2 <= nkx; ikx += 2) {
                        const size_t k = (size_t)iky * nkx + ikx;
                        const size_t idx_full = slab_offset + z_off + k;

                        float kx0 = (float)gr->kx[ikx];
                        float kx1 = (float)gr->kx[ikx + 1];
                        float wd0 = wd_y + wd_x * kx0;
                        float wd1 = wd_y + wd_x * kx1;

                        tgk_vc2 h0 = tgk_vc2_load(hz_0 + ikx);
                        float j0_0 = jz_0[ikx], j0_1 = jz_0[ikx + 1];

                        tgk_vc2 p0 = tgk_vc2_load(pz_0 + ikx);
                        if (has_apar) {
                            p0 = vfmsq_n_f32(p0, tgk_vc2_load(apar_z + k), v_fac);
                        }
                        tgk_vc2 chi0 = tgk_vc2_mul_real2(j0_0, j0_1, p0);

                        tgk_vc2 g0  = tgk_vc2_fma_real(h0, c_zt_f0, chi0);
                        tgk_vc2 rhs = tgk_vc2_mul_is2(w_star, w_star, chi0);
                        rhs = tgk_vc2_fma_is2(rhs, -wd0, -wd1, g0);

                        if (!skip_par_stream) {
                            tgk_vc2 hm2 = tgk_vc2_load(hz_m2 + ikx);
                            tgk_vc2 hm1 = tgk_vc2_load(hz_m1 + ikx);
                            tgk_vc2 hp1 = tgk_vc2_load(hz_p1 + ikx);
                            tgk_vc2 hp2 = tgk_vc2_load(hz_p2 + ikx);

                            tgk_vc2 pm2 = tgk_vc2_load(pz_m2 + ikx);
                            tgk_vc2 pm1 = tgk_vc2_load(pz_m1 + ikx);
                            tgk_vc2 pp1 = tgk_vc2_load(pz_p1 + ikx);
                            tgk_vc2 pp2 = tgk_vc2_load(pz_p2 + ikx);
                            if (has_apar) {
                                pm2 = vfmsq_n_f32(pm2, tgk_vc2_load(ctx->apar + z_off - 2 * nplane + k), v_fac);
                                pm1 = vfmsq_n_f32(pm1, tgk_vc2_load(ctx->apar + z_off - 1 * nplane + k), v_fac);
                                pp1 = vfmsq_n_f32(pp1, tgk_vc2_load(ctx->apar + z_off + 1 * nplane + k), v_fac);
                                pp2 = vfmsq_n_f32(pp2, tgk_vc2_load(ctx->apar + z_off + 2 * nplane + k), v_fac);
                            }
                            tgk_vc2 cm2 = tgk_vc2_mul_real2(jz_m2[ikx], jz_m2[ikx + 1], pm2);
                            tgk_vc2 cm1 = tgk_vc2_mul_real2(jz_m1[ikx], jz_m1[ikx + 1], pm1);
                            tgk_vc2 cp1 = tgk_vc2_mul_real2(jz_p1[ikx], jz_p1[ikx + 1], pp1);
                            tgk_vc2 cp2 = tgk_vc2_mul_real2(jz_p2[ikx], jz_p2[ikx + 1], pp2);

                            tgk_vc2 gm2 = tgk_vc2_fma_real(hm2, c_zt_f0, cm2);
                            tgk_vc2 gm1 = tgk_vc2_fma_real(hm1, c_zt_f0, cm1);
                            tgk_vc2 gp1 = tgk_vc2_fma_real(hp1, c_zt_f0, cp1);
                            tgk_vc2 gp2 = tgk_vc2_fma_real(hp2, c_zt_f0, cp2);

                            tgk_vc2 dg_dz = vsubq_f32(gm2, gp2);
                            dg_dz = vfmaq_n_f32(dg_dz, vsubq_f32(gp1, gm1), 8.0f);
                            rhs = vfmaq_n_f32(rhs, dg_dz, c_stream);

                            tgk_vc2 d4h = vaddq_f32(hm2, hp2);
                            d4h = vfmsq_n_f32(d4h, vaddq_f32(hm1, hp1), 4.0f);
                            d4h = vfmaq_n_f32(d4h, h0, 6.0f);
                            rhs = vfmaq_n_f32(rhs, d4h, c_hypz);
                        }

                        if (has_mirror) {
                            tgk_vc2 vm2 = h_vm2 ? tgk_vc2_load(h_vm2 + z_off + k) : tgk_vc2_zero();
                            tgk_vc2 vm1 = h_vm1 ? tgk_vc2_load(h_vm1 + z_off + k) : tgk_vc2_zero();
                            tgk_vc2 vp1 = h_vp1 ? tgk_vc2_load(h_vp1 + z_off + k) : tgk_vc2_zero();
                            tgk_vc2 vp2 = h_vp2 ? tgk_vc2_load(h_vp2 + z_off + k) : tgk_vc2_zero();

                            tgk_vc2 dh_dv = vsubq_f32(vm2, vp2);
                            dh_dv = vfmaq_n_f32(dh_dv, vsubq_f32(vp1, vm1), 8.0f);
                            rhs = vfmaq_n_f32(rhs, dh_dv, c_mirror);
                        }

                        if (do_nl) {
                            rhs = vaddq_f32(rhs, tgk_vc2_load(spec_tmp1 + k));
                        }

                        if (iky == 0 && ikx == 0) {
                            rhs = vsetq_lane_f32(0.0f, rhs, 0);
                            rhs = vsetq_lane_f32(0.0f, rhs, 1);
                        }

                        tgk_vc2 h_next;
                        if (stage_idx == 0) {
                            tgk_vc2 r_acc = vmulq_n_f32(rhs, dt_b);
                            tgk_vc2_store(ctx->h_rhs + idx_full, r_acc);
                            h_next = vfmaq_n_f32(h0, rhs, dt_c);
                            tgk_vc2_store(ctx->h_stage + idx_full, h_next);
                        } else if (!is_last_stg) {
                            tgk_vc2 r_acc = vfmaq_n_f32(tgk_vc2_load(ctx->h_rhs + idx_full), rhs, dt_b);
                            tgk_vc2_store(ctx->h_rhs + idx_full, r_acc);
                            h_next = vfmaq_n_f32(tgk_vc2_load(ctx->h + idx_full), rhs, dt_c);
                            tgk_vc2_store(ctx->h_stage + idx_full, h_next);
                        } else {
                            tgk_vc2 r_acc  = tgk_vc2_load(ctx->h_rhs + idx_full);
                            tgk_vc2 h_base = tgk_vc2_load(ctx->h     + idx_full);
                            h_next = vfmaq_n_f32(vaddq_f32(h_base, r_acc), rhs, dt_b);
                            tgk_vc2_store(ctx->h + idx_full, h_next);
                        }

                        double h0_re = (double)vgetq_lane_f32(h_next, 0);
                        double h0_im = (double)vgetq_lane_f32(h_next, 1);
                        double h1_re = (double)vgetq_lane_f32(h_next, 2);
                        double h1_im = (double)vgetq_lane_f32(h_next, 3);
                        double j0_d0 = (double)j0_0;
                        double j0_d1 = (double)j0_1;

                        qn_z[k].re     += wq * j0_d0 * h0_re;
                        qn_z[k].im     += wq * j0_d0 * h0_im;
                        qn_z[k + 1].re += wq * j0_d1 * h1_re;
                        qn_z[k + 1].im += wq * j0_d1 * h1_im;
                        if (has_apar) {
                            am_z[k].re     += wa * j0_d0 * h0_re;
                            am_z[k].im     += wa * j0_d0 * h0_im;
                            am_z[k + 1].re += wa * j0_d1 * h1_re;
                            am_z[k + 1].im += wa * j0_d1 * h1_im;
                        }
                    }
                }
#endif
                for (; ikx < nkx; ikx++) {
                    const size_t k = (size_t)iky * nkx + ikx;
                    const size_t idx_full = slab_offset + z_off + k;

                    const double kx = gr->kx[ikx];
                    const tgk_real w_d = (tgk_real)(v_drift * (cv_y * ky + cv_x * kx));

                    const tgk_complex h0 = h_slab[z_off + k];
                    const tgk_real j0_val = j0_z[k];

                    tgk_complex chi0 = phi_z[k];
                    if (has_apar) {
                        chi0.re -= v_fac * apar_z[k].re;
                        chi0.im -= v_fac * apar_z[k].im;
                    }
                    chi0 = tgk_cscale(j0_val, chi0);

                    /* 1. Linear drive & magnetic drift:
                     * rhs = i * w_star * chi0 - i * w_d * (h0 + (Z_s/T_s)*F0 * chi0) */
                    tgk_complex g0 = {h0.re + c_zt_f0 * chi0.re, h0.im + c_zt_f0 * chi0.im};
                    tgk_complex rhs = tgk_cmul_is(w_star, chi0);
                    tgk_complex wd_term = tgk_cmul_is(-w_d, g0);
                    rhs = tgk_cadd(rhs, wd_term);

                    /* 2. Parallel streaming & fourth-order z hyperdiffusion with twist-and-shift */
                    if (!skip_par_stream) {
                        tgk_complex hm2 = fetch_z_ts(gr, h_slab, iz - 2, iky, ikx);
                        tgk_complex hm1 = fetch_z_ts(gr, h_slab, iz - 1, iky, ikx);
                        tgk_complex hp1 = fetch_z_ts(gr, h_slab, iz + 1, iky, ikx);
                        tgk_complex hp2 = fetch_z_ts(gr, h_slab, iz + 2, iky, ikx);

                        tgk_complex cm2 = fetch_chi_ts(gr, ctx->phi, ctx->apar, j0_sm, v_fac, has_apar, iz - 2, iky, ikx);
                        tgk_complex cm1 = fetch_chi_ts(gr, ctx->phi, ctx->apar, j0_sm, v_fac, has_apar, iz - 1, iky, ikx);
                        tgk_complex cp1 = fetch_chi_ts(gr, ctx->phi, ctx->apar, j0_sm, v_fac, has_apar, iz + 1, iky, ikx);
                        tgk_complex cp2 = fetch_chi_ts(gr, ctx->phi, ctx->apar, j0_sm, v_fac, has_apar, iz + 2, iky, ikx);

                        /* g = h + (Z_s/T_s)*F0*chi along z */
                        tgk_complex gm2 = {hm2.re + c_zt_f0 * cm2.re, hm2.im + c_zt_f0 * cm2.im};
                        tgk_complex gm1 = {hm1.re + c_zt_f0 * cm1.re, hm1.im + c_zt_f0 * cm1.im};
                        tgk_complex gp1 = {hp1.re + c_zt_f0 * cp1.re, hp1.im + c_zt_f0 * cp1.im};
                        tgk_complex gp2 = {hp2.re + c_zt_f0 * cp2.re, hp2.im + c_zt_f0 * cp2.im};

                        /* 4th-order centered derivative: (-gp2 + 8*gp1 - 8*gm1 + gm2) / (12*dz) */
                        tgk_complex dg_dz = {
                            -gp2.re + 8.0f * gp1.re - 8.0f * gm1.re + gm2.re,
                            -gp2.im + 8.0f * gp1.im - 8.0f * gm1.im + gm2.im
                        };
                        rhs.re += c_stream * dg_dz.re;
                        rhs.im += c_stream * dg_dz.im;

                        /* 4th-order parallel hyperdiffusion: -(hm2 - 4*hm1 + 6*h0 - 4*hp1 + hp2) */
                        tgk_complex d4h = {
                            hm2.re - 4.0f * hm1.re + 6.0f * h0.re - 4.0f * hp1.re + hp2.re,
                            hm2.im - 4.0f * hm1.im + 6.0f * h0.im - 4.0f * hp1.im + hp2.im
                        };
                        rhs.re += c_hypz * d4h.re;
                        rhs.im += c_hypz * d4h.im;
                    }

                    /* 3. Mirror force term along v_parallel (4th-order centered finite difference) */
                    if (fabs(c_mirror) > 1e-18f) {
                        tgk_complex v_m2 = h_vm2 ? h_vm2[z_off + k] : (tgk_complex){0.0f, 0.0f};
                        tgk_complex v_m1 = h_vm1 ? h_vm1[z_off + k] : (tgk_complex){0.0f, 0.0f};
                        tgk_complex v_p1 = h_vp1 ? h_vp1[z_off + k] : (tgk_complex){0.0f, 0.0f};
                        tgk_complex v_p2 = h_vp2 ? h_vp2[z_off + k] : (tgk_complex){0.0f, 0.0f};

                        tgk_complex dh_dv = {
                            -v_p2.re + 8.0f * v_p1.re - 8.0f * v_m1.re + v_m2.re,
                            -v_p2.im + 8.0f * v_p1.im - 8.0f * v_m1.im + v_m2.im
                        };
                        rhs.re += c_mirror * dh_dv.re;
                        rhs.im += c_mirror * dh_dv.im;
                    }

                    /* 4. Nonlinear E x B term */
                    if (do_nl) {
                        rhs.re += spec_tmp1[k].re;
                        rhs.im += spec_tmp1[k].im;
                    }

                    /* Zero out (kx=0, ky=0) constant offset mode */
                    if (iky == 0 && ikx == 0) {
                        rhs = (tgk_complex){0.0f, 0.0f};
                    }

                    /* 5. Fused RK4 stage update AND next-stage field integral accumulation! */
                    tgk_complex h_next;
                    if (stage_idx == 0) {
                        tgk_complex r_acc = tgk_cscale(dt_b, rhs);
                        ctx->h_rhs[idx_full] = r_acc;
                        h_next.re = h0.re + dt_c * rhs.re;
                        h_next.im = h0.im + dt_c * rhs.im;
                        ctx->h_stage[idx_full] = h_next;
                    } else if (!is_last_stg) {
                        tgk_complex r_acc = ctx->h_rhs[idx_full];
                        r_acc.re += dt_b * rhs.re;
                        r_acc.im += dt_b * rhs.im;
                        ctx->h_rhs[idx_full] = r_acc;

                        tgk_complex h_base = ctx->h[idx_full];
                        h_next.re = h_base.re + dt_c * rhs.re;
                        h_next.im = h_base.im + dt_c * rhs.im;
                        ctx->h_stage[idx_full] = h_next;
                    } else {
                        /* Final RK4 stage (stage 3): write completed step directly into ctx->h */
                        tgk_complex r_acc  = ctx->h_rhs[idx_full];
                        tgk_complex h_base = ctx->h[idx_full];
                        h_next.re = h_base.re + r_acc.re + dt_b * rhs.re;
                        h_next.im = h_base.im + r_acc.im + dt_b * rhs.im;
                        ctx->h[idx_full] = h_next;
                    }

                    /* Accumulate h_next into thread-local FP64 field integrals for the NEXT stage */
                    double j0_d = (double)j0_val;
                    qn_z[k].re += wq * j0_d * (double)h_next.re;
                    qn_z[k].im += wq * j0_d * (double)h_next.im;
                    if (has_apar) {
                        am_z[k].re += wa * j0_d * (double)h_next.re;
                        am_z[k].im += wa * j0_d * (double)h_next.im;
                    }
                }
            }
        }
        }
    }

    ws->vmax_nl = vmax_nl_local;
}

/* Compute maximum linear CFL frequency across the grid (streaming + magnetic drifts + drive) */
double tgk_rhs_compute_linear_cfl(const tgk_ctx *ctx) {
    const tgk_grid  *gr = &ctx->grid;
    const tgk_geom  *gm = &ctx->geom;
    const tgk_param *p  = &ctx->param;

    double max_rate = 1e-6;
    double kx_max   = fabs(gr->kx[gr->nkx / 2]);
    double ky_max   = gr->ky[gr->nky - 1];
    double vmax     = fabs(gr->vpar[gr->nv - 1]);
    double mumax    = gr->mu[gr->nmu - 1];

    for (int s = 0; s < gr->nspecies; s++) {
        const tgk_species_param *sp = &p->spec[s];
        double vths = sqrt(sp->temp / sp->mass);
        int is_imex_elec = (p->imex && sp->charge < 0.0);

        for (int iz = 0; iz < gr->nz; iz++) {
            double b    = gm->bmag[iz];
            double gpar = fabs(gm->gradpar[iz]);
            double dbdz = fabs(gm->dbdz[iz]);

            double rate_stream = is_imex_elec ? 0.0 : (1.4 * vths * vmax * gpar / gm->dz);
            double rate_mirror = 1.4 * vths * mumax * dbdz * gpar / gr->dv;
            double rate_drift  = (sp->temp / fabs(sp->charge)) * (vmax * vmax + mumax * b) *
                                 (fabs(gm->cv_y[iz]) * ky_max + fabs(gm->cv_x[iz]) * kx_max);
            double rate_drive  = ky_max * (fabs(sp->fprim) + 2.0 * fabs(sp->tprim)) / p->rmaj;

            double tot = rate_stream + rate_mirror + rate_drift + rate_drive;
            if (tot > max_rate) max_rate = tot;
        }
    }

    return max_rate;
}
