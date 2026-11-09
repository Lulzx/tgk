#include "tgk.h"
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Accurate Bessel J0(x) using series for small x and asymptotic / rational approx for large x */
double tgk_bessel_j0(double x) {
    return j0(fabs(x));
}

/* Modified Bessel I0(x) * exp(-x) = Gamma0(x) for gyrokinetic polarization density */
double tgk_gamma0(double b) {
    if (b <= 0.0) return 1.0;
    if (b < 12.0) {
        /* Power series for I0(b): sum_{k=0}^inf (b^2 / 4)^k / (k!)^2 */
        double term = 1.0;
        double sum  = 1.0;
        double b2_4 = 0.25 * b * b;
        for (int k = 1; k < 60; k++) {
            term *= b2_4 / ((double)k * (double)k);
            sum  += term;
            if (term < 1e-16 * sum) break;
        }
        return exp(-b) * sum;
    } else {
        /* Asymptotic expansion for exp(-b) * I0(b) at large b */
        double inv8b = 1.0 / (8.0 * b);
        double s = 1.0 + inv8b * (1.0 + inv8b * (4.5 + inv8b * (37.5 + inv8b * 464.0625)));
        return s / sqrt(2.0 * M_PI * b);
    }
}

/* Compute n-point Gauss-Laguerre quadrature nodes x_i and weights w_i for int_0^inf exp(-x) f(x) dx
 * via Golub-Welsch tridiagonal Jacobi matrix algorithm (exact for polynomials up to 2n-1). */
static void compute_gauss_laguerre(int n, double *x, double *w) {
    /* Symmetric tridiagonal matrix: diagonal a_i = 2*i + 1, off-diagonal b_i = i (for i=1..n-1) */
    double *d = calloc(n, sizeof(double));
    double *e = calloc(n, sizeof(double));
    double *z = calloc(n * n, sizeof(double));

    for (int i = 0; i < n; i++) {
        d[i] = 2.0 * i + 1.0;
        e[i] = (double)(i + 1);
        z[i * n + i] = 1.0;
    }
    e[n - 1] = 0.0;

    /* Implicit QL algorithm with Wilkinson shift */
    for (int l = 0; l < n; l++) {
        int iter = 0;
        int m;
        do {
            for (m = l; m < n - 1; m++) {
                double dd = fabs(d[m]) + fabs(d[m + 1]);
                if (fabs(e[m]) + dd == dd) break;
            }
            if (m != l) {
                if (++iter > 100) break;
                double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
                double r = hypot(g, 1.0);
                g = d[m] - d[l] + e[l] / (g + (g >= 0.0 ? fabs(r) : -fabs(r)));
                double s = 1.0, c = 1.0, p = 0.0;
                for (int i = m - 1; i >= l; i--) {
                    double f = s * e[i];
                    double b = c * e[i];
                    r = hypot(f, g);
                    e[i + 1] = r;
                    if (r == 0.0) {
                        d[i + 1] -= p;
                        e[m] = 0.0;
                        break;
                    }
                    s = f / r;
                    c = g / r;
                    g = d[i + 1] - p;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    p = s * r;
                    d[i + 1] = g + p;
                    g = c * r - b;
                    for (int k = 0; k < n; k++) {
                        double fz = z[k * n + i + 1];
                        z[k * n + i + 1] = s * z[k * n + i] + c * fz;
                        z[k * n + i]     = c * z[k * n + i] - s * fz;
                    }
                }
                if (r == 0.0 && iter <= 100) continue;
                d[l] -= p;
                e[l] = g;
                e[m] = 0.0;
            }
        } while (m != l);
    }

    /* Sort eigenvalues ascending */
    for (int i = 0; i < n - 1; i++) {
        int k = i;
        double p = d[i];
        for (int j = i + 1; j < n; j++) {
            if (d[j] < p) {
                k = j;
                p = d[j];
            }
        }
        if (k != i) {
            d[k] = d[i];
            d[i] = p;
            for (int j = 0; j < n; j++) {
                double tmp = z[j * n + i];
                z[j * n + i] = z[j * n + k];
                z[j * n + k] = tmp;
            }
        }
    }

    for (int i = 0; i < n; i++) {
        x[i] = d[i];
        w[i] = z[0 * n + i] * z[0 * n + i]; /* sum(w) = 1.0 */
    }

    free(d);
    free(e);
    free(z);
}

int tgk_grid_init(const tgk_param *p, const tgk_geom *g, tgk_grid *gr) {
    memset(gr, 0, sizeof(*gr));
    gr->nkx      = p->nkx;
    gr->nky      = p->nky;
    gr->nz       = p->nz;
    gr->nv       = p->nv;
    gr->nmu      = p->nmu;
    gr->nspecies = p->nspecies;

    gr->nplane = (size_t)gr->nky * (size_t)gr->nkx;
    gr->nfield = (size_t)gr->nz * gr->nplane;
    gr->nslab  = gr->nfield;
    gr->ntot   = (size_t)gr->nspecies * (size_t)gr->nmu * (size_t)gr->nv * gr->nslab;

    gr->kx   = tgk_malloc_aligned(sizeof(double) * gr->nkx);
    gr->ky   = tgk_malloc_aligned(sizeof(double) * gr->nky);
    gr->vpar = tgk_malloc_aligned(sizeof(double) * gr->nv);
    gr->w_v  = tgk_malloc_aligned(sizeof(double) * gr->nv);
    gr->mu   = tgk_malloc_aligned(sizeof(double) * gr->nmu);
    gr->w_mu = tgk_malloc_aligned(sizeof(double) * gr->nmu);

    /* 1. Perpendicular wavenumbers ky and kx */
    double dky = (p->ky_min > 0.0) ? p->ky_min : 0.05;
    for (int iky = 0; iky < gr->nky; iky++) {
        gr->ky[iky] = (double)iky * dky;
    }

    /* Twist-and-shift quantized dkx = 2*pi * |shat| * dky / N_twist, or specified by kx_max */
    double dkx;
    if (p->kx_max > 0.0 && gr->nkx > 1) {
        dkx = p->kx_max / (double)(gr->nkx / 2);
    } else if (fabs(p->shat) > 1e-8) {
        dkx = 2.0 * M_PI * fabs(p->shat) * dky * p->z_Period;
        if (dkx > 0.2) dkx *= 0.25; /* finer radial resolution if 2*pi*shat*dky is coarse */
    } else {
        dkx = dky;
    }

    for (int ikx = 0; ikx < gr->nkx; ikx++) {
        if (ikx <= gr->nkx / 2) {
            gr->kx[ikx] = (double)ikx * dkx;
        } else {
            gr->kx[ikx] = (double)(ikx - gr->nkx) * dkx;
        }
    }

    /* 2. Parallel velocity v_parallel grid in [-vmax, vmax] */
    double vmax = (p->vmax > 0.0) ? p->vmax : 3.5;
    gr->dv = (gr->nv > 1) ? (2.0 * vmax / (double)(gr->nv - 1)) : 1.0;
    double v_norm = 0.0;
    for (int iv = 0; iv < gr->nv; iv++) {
        gr->vpar[iv] = -vmax + (double)iv * gr->dv;
        /* Composite Simpson / spectral-accurate trapezoidal rule on [-vmax, vmax] */
        double w = gr->dv;
        if (iv == 0 || iv == gr->nv - 1) w *= 0.5;
        gr->w_v[iv] = w;
        v_norm += w * exp(-0.5 * gr->vpar[iv] * gr->vpar[iv]) / sqrt(2.0 * M_PI);
    }
    /* Normalize w_v so int dv F_M(vpar) = 1.0 to machine precision on discrete grid */
    for (int iv = 0; iv < gr->nv; iv++) {
        gr->w_v[iv] /= v_norm;
    }

    /* 3. Magnetic moment mu grid via Gauss-Laguerre quadrature */
    compute_gauss_laguerre(gr->nmu, gr->mu, gr->w_mu);

    /* 4. Equilibrium Maxwellian F0 and velocity quadrature weights w_vel */
    size_t n_smvz = (size_t)gr->nspecies * gr->nmu * gr->nv * gr->nz;
    gr->f0    = tgk_malloc_aligned(sizeof(tgk_real) * n_smvz);
    gr->w_vel = tgk_malloc_aligned(sizeof(double) * n_smvz);

    for (int s = 0; s < gr->nspecies; s++) {
        for (int imu = 0; imu < gr->nmu; imu++) {
            double mu = gr->mu[imu];
            double wmu_exp = gr->w_mu[imu] * exp(mu); /* undo exp(-mu) weight for local B(z) */
            for (int iv = 0; iv < gr->nv; iv++) {
                double v = gr->vpar[iv];
                double fv = exp(-0.5 * v * v) / sqrt(2.0 * M_PI);
                double wv = gr->w_v[iv];
                for (int iz = 0; iz < gr->nz; iz++) {
                    double b = g->bmag[iz];
                    /* F0(vpar, mu, z) normalized so int d^3v F0 = 1 at every z:
                     * d^3v = 2*pi * B(z) dmu dvpar. With mu in units of T_s/B_ref:
                     * int_0^inf B(z) exp(-mu * B(z)) dmu = 1. */
                    double f_mu_z = exp(-mu * b);
                    size_t idx = (((size_t)s * gr->nmu + imu) * gr->nv + iv) * gr->nz + iz;
                    gr->f0[idx]    = (tgk_real)(fv * f_mu_z);
                    gr->w_vel[idx] = b * wmu_exp * wv;
                }
            }
        }
    }

    /* Normalize w_vel at each (s, iz) so sum_{imu, iv} w_vel * f0 = 1.0 to machine precision */
    for (int s = 0; s < gr->nspecies; s++) {
        for (int iz = 0; iz < gr->nz; iz++) {
            double sum = 0.0;
            for (int imu = 0; imu < gr->nmu; imu++) {
                for (int iv = 0; iv < gr->nv; iv++) {
                    size_t idx = (((size_t)s * gr->nmu + imu) * gr->nv + iv) * gr->nz + iz;
                    sum += gr->w_vel[idx] * (double)gr->f0[idx];
                }
            }
            if (sum > 0.0) {
                for (int imu = 0; imu < gr->nmu; imu++) {
                    for (int iv = 0; iv < gr->nv; iv++) {
                        size_t idx = (((size_t)s * gr->nmu + imu) * gr->nv + iv) * gr->nz + iz;
                        gr->w_vel[idx] /= sum;
                    }
                }
            }
        }
    }

    /* 5. Precompute k_perp^2, Bessel J0(k_perp * rho_s), and Quasineutrality / Ampere denominators */
    size_t n_smf = (size_t)gr->nspecies * gr->nmu * gr->nfield;
    gr->j0        = tgk_malloc_aligned(sizeof(tgk_real) * n_smf);
    gr->qn_denom  = tgk_malloc_aligned(sizeof(double) * gr->nfield);
    gr->zf_fac    = tgk_malloc_aligned(sizeof(double) * gr->nkx);
    gr->amp_denom = tgk_malloc_aligned(sizeof(double) * gr->nfield);
    gr->kperp2    = tgk_malloc_aligned(sizeof(double) * gr->nfield);

    for (int iz = 0; iz < gr->nz; iz++) {
        double b   = g->bmag[iz];
        double gxx = g->gxx[iz];
        double gxy = g->gxy[iz];
        double gyy = g->gyy[iz];

        for (int iky = 0; iky < gr->nky; iky++) {
            double ky = gr->ky[iky];
            for (int ikx = 0; ikx < gr->nkx; ikx++) {
                double kx = gr->kx[ikx];
                double kp2 = gxx * kx * kx + 2.0 * gxy * kx * ky + gyy * ky * ky;
                if (kp2 < 0.0) kp2 = 0.0;
                double kp = sqrt(kp2);

                size_t fidx = ((size_t)iz * gr->nky + iky) * gr->nkx + ikx;
                gr->kperp2[fidx] = kp2;

                /* Gyroaveraging J0(k_perp * v_perp / Omega_s)
                 * With v_perp = v_th_s * sqrt(2 * mu * B(z)) and rho_s = v_th_s / Omega_s:
                 * k_perp * rho_s(mu) = kp * (sqrt(m_s * T_s) / (|Z_s| * B(z))) * sqrt(2 * mu * B(z)) */
                for (int s = 0; s < gr->nspecies; s++) {
                    double zs = fabs(p->spec[s].charge);
                    if (zs < 1e-12) zs = 1.0;
                    double rho_th = sqrt(p->spec[s].mass * p->spec[s].temp) / (zs * b);
                    for (int imu = 0; imu < gr->nmu; imu++) {
                        double arg = kp * rho_th * sqrt(2.0 * gr->mu[imu] * b);
                        size_t jidx = ((size_t)s * gr->nmu + imu) * gr->nfield + fidx;
                        gr->j0[jidx] = (tgk_real)tgk_bessel_j0(arg);
                    }
                }

                /* Quasineutrality polarization denominator:
                 * sum_s (Z_s^2 n_s / T_s) * (1 - Gamma_0(b_s)) + adiabatic electron term */
                double qn_d = 0.0;
                double amp_d = kp2;

                for (int s = 0; s < gr->nspecies; s++) {
                    double zs   = p->spec[s].charge;
                    double ns   = p->spec[s].dens;
                    double ts   = p->spec[s].temp;
                    double ms   = p->spec[s].mass;
                    /* Compute discrete velocity integral of J0^2 to be 100% consistent with numerator */
                    double gamma0_disc = 0.0;
                    for (int imu = 0; imu < gr->nmu; imu++) {
                        size_t jidx = ((size_t)s * gr->nmu + imu) * gr->nfield + fidx;
                        double j0_val = (double)gr->j0[jidx];
                        for (int iv = 0; iv < gr->nv; iv++) {
                            size_t vidx = (((size_t)s * gr->nmu + imu) * gr->nv + iv) * gr->nz + iz;
                            gamma0_disc += gr->w_vel[vidx] * (double)gr->f0[vidx] * j0_val * j0_val;
                        }
                    }
                    qn_d += (zs * zs * ns / ts) * (1.0 - gamma0_disc);

                    if (p->beta > 0.0) {
                        /* Parallel Ampere's law skin-depth / beta term when evolving g_s = h_s + ... */
                        amp_d += (p->beta * zs * zs * ns / ms) * gamma0_disc;
                    }
                }

                if (p->adiabatic_electrons && iky > 0) {
                    /* Non-zonal modes include adiabatic electron response (phi - <phi>_FS) */
                    qn_d += p->tau_e;
                }

                gr->qn_denom[fidx]  = qn_d;
                gr->amp_denom[fidx] = amp_d;
            }
        }
    }

    /* Precompute zonal-flow flux-surface-averaged correction factor for adiabatic electrons:
     * At ky = 0: [Q_pol(kx, z) + tau_e] * phi(kx, z) - tau_e * <phi(kx, z)>_FS = N(kx, z)
     * Let D_pol(kx, z) = qn_denom(kx, ky=0, z) and D_tot(kx, z) = D_pol(kx, z) + tau_e.
     * Then phi(kx, z) = N(kx, z) / D_tot(kx, z) + tau_e * <phi>_FS / D_tot(kx, z).
     * Taking <...>_FS gives: <phi>_FS * (1 - tau_e * <1 / D_tot>_FS) = <N / D_tot>_FS. */
    for (int ikx = 0; ikx < gr->nkx; ikx++) {
        if (!p->adiabatic_electrons || ikx == 0) {
            gr->zf_fac[ikx] = 0.0;
            continue;
        }
        double avg_inv_dtot = 0.0;
        for (int iz = 0; iz < gr->nz; iz++) {
            size_t fidx = ((size_t)iz * gr->nky + 0) * gr->nkx + ikx;
            double d_pol = gr->qn_denom[fidx];
            double d_tot = d_pol + p->tau_e;
            avg_inv_dtot += g->z_weight[iz] / d_tot;
        }
        double denom_zf = 1.0 - p->tau_e * avg_inv_dtot;
        gr->zf_fac[ikx] = (fabs(denom_zf) > 1e-12) ? (p->tau_e / denom_zf) : 0.0;
    }

    /* 6. Twist-and-shift boundary condition lookup tables:
     * Across the flux-tube ends z = -pi*N and z = +pi*N, magnetic shear shifts radial modes by
     * delta_kx = 2*pi * N * shat * ky.
     * For ky = 0 (zonal modes), delta_kx = 0 (strictly periodic in z). */
    gr->ts_minus = tgk_malloc_aligned(sizeof(int) * gr->nplane);
    gr->ts_plus  = tgk_malloc_aligned(sizeof(int) * gr->nplane);

    for (int iky = 0; iky < gr->nky; iky++) {
        double ky = gr->ky[iky];
        double shift_kx = 2.0 * M_PI * p->z_Period * p->shat * ky;
        int di = (dkx > 1e-12) ? (int)llround(shift_kx / dkx) : 0;

        for (int ikx = 0; ikx < gr->nkx; ikx++) {
            size_t pidx = (size_t)iky * gr->nkx + ikx;
            if (iky == 0 || di == 0) {
                /* Periodic connection for ky=0 or zero shear */
                gr->ts_minus[pidx] = ikx;
                gr->ts_plus[pidx]  = ikx;
            } else {
                /* Map ikx to logical signed index m in [-nkx/2+1, nkx/2] */
                int m = (ikx <= gr->nkx / 2) ? ikx : (ikx - gr->nkx);
                int m_minus = m - di;
                int m_plus  = m + di;
                int half = gr->nkx / 2;

                if (m_minus >= -half + 1 && m_minus <= half) {
                    gr->ts_minus[pidx] = (m_minus >= 0) ? m_minus : (m_minus + gr->nkx);
                } else {
                    gr->ts_minus[pidx] = -1; /* Absorbing boundary at end of ballooning chain */
                }

                if (m_plus >= -half + 1 && m_plus <= half) {
                    gr->ts_plus[pidx] = (m_plus >= 0) ? m_plus : (m_plus + gr->nkx);
                } else {
                    gr->ts_plus[pidx] = -1;
                }
            }
        }
    }

    return 0;
}

void tgk_grid_free(tgk_grid *gr) {
    tgk_free_aligned(gr->kx);
    tgk_free_aligned(gr->ky);
    tgk_free_aligned(gr->vpar);
    tgk_free_aligned(gr->w_v);
    tgk_free_aligned(gr->mu);
    tgk_free_aligned(gr->w_mu);
    tgk_free_aligned(gr->f0);
    tgk_free_aligned(gr->w_vel);
    tgk_free_aligned(gr->j0);
    tgk_free_aligned(gr->qn_denom);
    tgk_free_aligned(gr->zf_fac);
    tgk_free_aligned(gr->amp_denom);
    tgk_free_aligned(gr->kperp2);
    tgk_free_aligned(gr->ts_minus);
    tgk_free_aligned(gr->ts_plus);
    memset(gr, 0, sizeof(*gr));
}
