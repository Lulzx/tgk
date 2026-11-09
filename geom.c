#include "tgk.h"
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int tgk_geom_init(const tgk_param *p, tgk_geom *g) {
    int nz = p->nz;
    if (nz < 4) return -1;

    memset(g, 0, sizeof(*g));
    g->z        = tgk_malloc_aligned(sizeof(double) * nz);
    g->bmag     = tgk_malloc_aligned(sizeof(double) * nz);
    g->dbdz     = tgk_malloc_aligned(sizeof(double) * nz);
    g->gradpar  = tgk_malloc_aligned(sizeof(double) * nz);
    g->gxx      = tgk_malloc_aligned(sizeof(double) * nz);
    g->gxy      = tgk_malloc_aligned(sizeof(double) * nz);
    g->gyy      = tgk_malloc_aligned(sizeof(double) * nz);
    g->cv_x     = tgk_malloc_aligned(sizeof(double) * nz);
    g->cv_y     = tgk_malloc_aligned(sizeof(double) * nz);
    g->gb_x     = tgk_malloc_aligned(sizeof(double) * nz);
    g->gb_y     = tgk_malloc_aligned(sizeof(double) * nz);
    g->jac      = tgk_malloc_aligned(sizeof(double) * nz);
    g->z_weight = tgk_malloc_aligned(sizeof(double) * nz);

    double z_max = M_PI * p->z_Period;
    g->dz = (2.0 * z_max) / (double)nz;

    double q0    = p->q0;
    double shat  = p->shat;
    double eps   = p->eps;
    double rmaj  = p->rmaj;
    double alpha = p->alpha_mhd;

    double w_sum = 0.0;

    for (int iz = 0; iz < nz; iz++) {
        /* Cell-centered uniform poloidal angle grid in [-pi*N, pi*N) */
        double z = -z_max + ((double)iz + 0.5) * g->dz;
        g->z[iz] = z;

        if (p->geom_type == TGK_GEOM_S_ALPHA) {
            /* Standard analytic s-alpha local equilibrium model (Dimits et al. 2000) */
            double cosz = cos(z);
            double sinz = sin(z);
            double b = 1.0 / (1.0 + eps * cosz);
            g->bmag[iz]    = b;
            g->dbdz[iz]    = eps * sinz * b * b;
            g->gradpar[iz] = 1.0 / (q0 * rmaj);

            double lambda = shat * z - alpha * sinz;
            g->gxx[iz] = 1.0;
            g->gxy[iz] = lambda;
            g->gyy[iz] = 1.0 + lambda * lambda;

            /* Magnetic drift coefficients in normalized units (v_th / R0):
             * omega_d = (v_par^2 + mu*B) * (cv_y * ky + cv_x * kx)
             * In standard s-alpha: omega_d_y = -(cos(z) + (shat*z - alpha*sin(z))*sin(z)) / R0 */
            double d_y = -(cosz + lambda * sinz) / rmaj;
            double d_x = -(sinz) / rmaj;

            g->cv_y[iz] = d_y;
            g->cv_x[iz] = d_x;
            g->gb_y[iz] = d_y;
            g->gb_x[iz] = d_x;

            g->jac[iz] = 1.0 / (b * g->gradpar[iz]);
        } else {
            /* Miller local shaped equilibrium parameterization (Miller et al. PoP 1998):
             * R(r, z) = R0(r) + r * cos(z + arcsin(delta) * sin(z))
             * Z(r, z) = kappa * r * sin(z) */
            double kappa   = (p->kappa > 0.0) ? p->kappa : 1.0;
            double s_kappa = p->s_kappa;
            double delta   = p->delta;
            double s_delta = p->s_delta;
            double shift   = p->shift;

            double x_asin = asin(fmin(fmax(delta, -0.99), 0.99));
            double theta_r = z + x_asin * sin(z);
            double dtheta_r_dz = 1.0 + x_asin * cos(z);

            double cos_tr = cos(theta_r);
            double sin_tr = sin(theta_r);
            double cosz   = cos(z);
            double sinz   = sin(z);

            /* Radial & poloidal derivatives of (R, Z) normalized by minor radius a */
            double dR_dr = shift + cos_tr - s_delta * sinz * sin_tr;
            double dR_dz = -sin_tr * dtheta_r_dz;
            double dZ_dr = kappa * (1.0 + s_kappa) * sinz;
            double dZ_dz = kappa * cosz;

            double det = dR_dr * dZ_dz - dR_dz * dZ_dr;
            if (fabs(det) < 1e-8) det = 1.0;

            double grad_r2 = (dR_dz * dR_dz + dZ_dz * dZ_dz) / (det * det);
            double grad_r_theta = -(dR_dr * dR_dz + dZ_dr * dZ_dz) / (det * det);

            double R_norm = 1.0 + eps * cos_tr;
            double b_tor = 1.0 / R_norm;
            double b_pol = (eps / q0) * sqrt(grad_r2) / R_norm;
            double b = sqrt(b_tor * b_tor + b_pol * b_pol);

            g->bmag[iz] = b;
            g->gradpar[iz] = 1.0 / (q0 * rmaj * det / kappa);

            /* Local integrated shear with shaping correction */
            double shat_eff = shat * z - alpha * sinz;
            g->gxx[iz] = grad_r2;
            g->gxy[iz] = grad_r2 * shat_eff + grad_r_theta;
            g->gyy[iz] = (1.0 / grad_r2) + g->gxy[iz] * g->gxy[iz] / grad_r2;

            /* Drift components with Miller shaping factor */
            double d_y = -(cos_tr + shat_eff * sin_tr * dtheta_r_dz) / (rmaj * kappa);
            double d_x = -(sin_tr * dtheta_r_dz) / (rmaj * kappa);

            g->cv_y[iz] = d_y;
            g->cv_x[iz] = d_x;
            g->gb_y[iz] = d_y + (alpha / (2.0 * q0 * q0 * rmaj)) / (b * b);
            g->gb_x[iz] = d_x;

            g->jac[iz] = fabs(det) * R_norm;
        }

        w_sum += g->jac[iz];
    }

    /* Compute dbdz numerically if Miller geometry, and normalize flux-surface weights */
    for (int iz = 0; iz < nz; iz++) {
        g->z_weight[iz] = g->jac[iz] / w_sum;
        if (p->geom_type == TGK_GEOM_MILLER) {
            int izm1 = (iz - 1 + nz) % nz;
            int izp1 = (iz + 1) % nz;
            g->dbdz[iz] = (g->bmag[izp1] - g->bmag[izm1]) / (2.0 * g->dz);
        }
    }

    return 0;
}

void tgk_geom_free(tgk_geom *g) {
    tgk_free_aligned(g->z);
    tgk_free_aligned(g->bmag);
    tgk_free_aligned(g->dbdz);
    tgk_free_aligned(g->gradpar);
    tgk_free_aligned(g->gxx);
    tgk_free_aligned(g->gxy);
    tgk_free_aligned(g->gyy);
    tgk_free_aligned(g->cv_x);
    tgk_free_aligned(g->cv_y);
    tgk_free_aligned(g->gb_x);
    tgk_free_aligned(g->gb_y);
    tgk_free_aligned(g->jac);
    tgk_free_aligned(g->z_weight);
    memset(g, 0, sizeof(*g));
}
