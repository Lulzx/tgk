#include "tgk.h"
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Geometry conventions (shared by both models)
 * --------------------------------------------
 * Lengths in a, B in B_ref, the parallel coordinate z is a poloidal angle with constant
 * gradpar = b.grad(z). With y = (dpsi/drho) alpha and x such that kx = shat * theta0 * ky,
 *     k_perp^2 = gxx kx^2 + 2 gxy kx ky + gyy ky^2,
 *     omega_d  = (T/Z) [ v_par^2 (cv_y ky + cv_x kx) + mu B (gb_y ky + gb_x kx) ].
 * In terms of the GS2 coefficients (gds2, gds21, gds22, cvdrift, gbdrift, cvdrift0, gbdrift0):
 *     gyy = gds2, gxy = -gds21 / shat, gxx = gds22 / shat^2,
 *     cv_y = -cvdrift / 2, gb_y = -gbdrift / 2, cv_x = cvdrift0 / (2 shat), gb_x = gbdrift0 / (2 shat).
 * (TGK's kx has the opposite sign to GS2's, which flips gxy and cv_x; the factor 1/2 is the
 * difference between GS2's sqrt(2T/m) and TGK's sqrt(T/m) drift normalization.) */

static double lerp_table(const double *xs, const double *ys, int n, double x) {
    /* xs ascending; clamp outside */
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (xs[mid] <= x) lo = mid; else hi = mid;
    }
    double f = (x - xs[lo]) / (xs[hi] - xs[lo]);
    return ys[lo] + f * (ys[hi] - ys[lo]);
}

/* Local Miller equilibrium (Miller et al., Phys. Plasmas 5, 973 (1998)):
 *     R(r, t) = R0(r) + r cos(t + arcsin(delta) sin t),   Z(r, t) = kappa r sin t,
 * with radial derivatives set by shift = dR0/dr, s_kappa = (r/kappa) dkappa/dr,
 * s_delta = r ddelta/dr. The radial derivative of the field-line label is obtained with
 * Bishop's method (as in GS2 bishop = 4 and GX's Miller module): the local Grad-Shafranov
 * equation removes second radial derivatives of the shape, and dF/dpsi is fixed so the
 * integrated shear reproduces shat. B_ref is the toroidal field at R_geo = R0, so F = R0.
 * The parallel coordinate is the equal-arc angle, so gradpar is constant. */
static int miller_init(const tgk_param *p, tgk_geom *g) {
    const int N = 8192;                       /* fine grid intervals over t in [0, 2 pi] */
    const double r     = p->eps * p->rmaj;
    const double R0    = p->rmaj;
    const double F     = p->rmaj;
    const double q     = p->q0;
    const double shat  = p->shat;
    const double kappa = (p->kappa > 0.0) ? p->kappa : 1.0;
    const double sk    = p->s_kappa;
    const double xd    = asin(fmin(fmax(p->delta, -0.99), 0.99));
    const double rxp   = p->s_delta / sqrt(1.0 - p->delta * p->delta);   /* r dx/dr */
    const double dt    = 2.0 * M_PI / N;

    if (fabs(shat) < 1e-6) {
        fprintf(stderr, "tgk: Miller geometry requires shat != 0\n");
        return -1;
    }

    size_t sz = sizeof(double) * (size_t)(N + 1);
    double *R = malloc(sz), *J = malloc(sz), *ell = malloc(sz), *gradr = malloc(sz);
    double *sinu = malloc(sz), *Rc = malloc(sz), *Bp = malloc(sz), *B = malloc(sz);
    double *nu = malloc(sz), *I1 = malloc(sz), *I2 = malloc(sz), *I3 = malloc(sz);
    double *D = malloc(sz), *Lpar = malloc(sz), *th_ea = malloc(sz), *dB2dl = malloc(sz);
    double *dBdr = malloc(sz);

    for (int j = 0; j <= N; j++) {
        double t  = j * dt;
        double a  = t + xd * sin(t);
        double ca = cos(a), sa = sin(a);
        double da = 1.0 + xd * cos(t);
        double d2a = -xd * sin(t);
        double Rt  = -r * sa * da;
        double Zt  = kappa * r * cos(t);
        double Rtt = -r * (ca * da * da + sa * d2a);
        double Ztt = -kappa * r * sin(t);
        double Rr  = p->shift + ca - rxp * sin(t) * sa;
        double Zr  = kappa * (1.0 + sk) * sin(t);
        R[j]     = R0 + r * ca;
        J[j]     = Rr * Zt - Rt * Zr;                       /* d(R,Z)/d(r,t) */
        ell[j]   = sqrt(Rt * Rt + Zt * Zt);                 /* dl/dt (poloidal arc length) */
        gradr[j] = ell[j] / J[j];                           /* |grad r| */
        sinu[j]  = Zt / ell[j];                             /* sin of tangent angle, l along t */
        Rc[j]    = ell[j] * ell[j] * ell[j] / (Rt * Ztt - Zt * Rtt);  /* curvature radius > 0 */
    }

    /* |dpsi/dr| from the safety factor: q = F / (2 pi psi') * oint J / R dt */
    double I_JR = 0.0;
    for (int j = 0; j < N; j++) I_JR += J[j] / R[j] * dt;
    const double psi_abs = F * I_JR / (2.0 * M_PI * q);
    const double psi_r   = -psi_abs;                        /* GS2/GX sign convention */
    const double pp      = 0.5 * p->betaprim / psi_r;       /* dp/dpsi, p in B_ref^2 / 4 pi */
    const double q_rho   = shat * q / r;
    const double qp_psi  = q_rho / psi_r;

    for (int j = 0; j <= N; j++) {
        Bp[j] = psi_abs * gradr[j] / R[j];
        B[j]  = sqrt(Bp[j] * Bp[j] + (F / R[j]) * (F / R[j]));
    }

    /* Field-line angle nu (= q theta_straight) and Bishop integrals, d nu = F J / (R psi') dt */
    nu[0] = I1[0] = I2[0] = I3[0] = Lpar[0] = 0.0;
    for (int j = 1; j <= N; j++) {
        double f0 = F * J[j - 1] / (R[j - 1] * psi_abs), f1 = F * J[j] / (R[j] * psi_abs);
        nu[j] = nu[j - 1] + 0.5 * dt * (f0 + f1);
        I1[j] = I1[j - 1] + 0.5 * dt * (f0 / (R[j-1] * R[j-1] * Bp[j-1] * Bp[j-1]) +
                                        f1 / (R[j] * R[j] * Bp[j] * Bp[j]));
        I2[j] = I2[j - 1] + 0.5 * dt * (f0 / (Bp[j-1] * Bp[j-1]) + f1 / (Bp[j] * Bp[j]));
        I3[j] = I3[j - 1] + 0.5 * dt * (f0 * (2.0 * sinu[j-1] / R[j-1] - 2.0 / Rc[j-1]) / (R[j-1] * Bp[j-1]) +
                                        f1 * (2.0 * sinu[j] / R[j] - 2.0 / Rc[j]) / (R[j] * Bp[j]));
        Lpar[j] = Lpar[j - 1] + 0.5 * dt * (B[j-1] / Bp[j-1] * ell[j-1] + B[j] / Bp[j] * ell[j]);
    }
    if (fabs(nu[N] / (2.0 * M_PI * q) - 1.0) > 1e-9) {
        fprintf(stderr, "tgk: Miller q integral inconsistent (%g)\n", nu[N] / (2.0 * M_PI));
    }

    /* D = grad(nu).grad(psi) / |grad psi|^2 = -(a_s F' + b_s p' - c_s) / 2, with
     * a_s = -(2 nu / F + 2 F I1), b_s = -2 I2, c_s = 2 I3, and F' such that D(2 pi) = 2 pi dq/dpsi. */
    double aN = -(2.0 * nu[N] / F + 2.0 * F * I1[N]);
    double Fp = (-2.0 * qp_psi * 2.0 * M_PI + 2.0 * I2[N] * pp + 2.0 * I3[N]) / aN;
    for (int j = 0; j <= N; j++) {
        double a_s = -(2.0 * nu[j] / F + 2.0 * F * I1[j]);
        double b_s = -2.0 * I2[j];
        double c_s =  2.0 * I3[j];
        D[j] = -0.5 * (a_s * Fp + b_s * pp - c_s);
        th_ea[j] = 2.0 * M_PI * Lpar[j] / Lpar[N];
    }

    for (int j = 0; j <= N; j++) {
        int jm = (j == 0) ? N - 1 : j - 1;
        int jp = (j == N) ? 1 : j + 1;
        dB2dl[j] = (B[jp] * B[jp] - B[jm] * B[jm]) / (2.0 * dt * ell[j]);
        dBdr[j]  = (Bp[j] / B[j]) * (-Bp[j] / Rc[j] + pp * R[j] - F * F * sinu[j] / (R[j] * R[j] * R[j] * Bp[j]));
    }

    const double gradpar = 2.0 * M_PI / Lpar[N];
    g->kxfac = fabs(q * psi_r / r);

    /* Interpolate onto the TGK z grid (equal-arc angle). Miller surfaces are up-down
     * symmetric: B, R, Bp, grad r, dB/dr are even in z, D and dB^2/dl are odd, and D
     * gains 2 pi dq/dpsi per poloidal turn. */
    for (int iz = 0; iz < p->nz; iz++) {
        double z  = g->z[iz];
        double nt = floor((z + M_PI) / (2.0 * M_PI));
        double th = z - 2.0 * M_PI * nt;                  /* in [-pi, pi) */
        double sg = (th < 0.0) ? -1.0 : 1.0;
        double ath = fabs(th);

        double Rz   = lerp_table(th_ea, R, N + 1, ath);
        double Bpz  = lerp_table(th_ea, Bp, N + 1, ath);
        double Bz   = lerp_table(th_ea, B, N + 1, ath);
        double grz  = lerp_table(th_ea, gradr, N + 1, ath);
        double dBr  = lerp_table(th_ea, dBdr, N + 1, ath);
        double dB2  = sg * lerp_table(th_ea, dB2dl, N + 1, ath);
        double Dz   = sg * lerp_table(th_ea, D, N + 1, ath) + nt * 2.0 * M_PI * qp_psi;

        double gpsi  = Rz * Bpz;                           /* |grad psi| */
        double gds2  = psi_r * psi_r * (Bz * Bz / (gpsi * gpsi) + Dz * Dz * gpsi * gpsi);
        double gds21 = -q_rho * psi_r * gpsi * gpsi * Dz;
        double gds22 = q_rho * q_rho * gpsi * gpsi;

        double aprime   = gpsi * Dz * psi_abs;
        double gbdrift  = (psi_abs / (Bz * Bz * Bz)) *
                          (2.0 * Bz * Bz * dBr / (-gpsi) + aprime * (1.0 / psi_r) * F / Rz * dB2 / Bz);
        double cvdrift  = (psi_abs / (Bz * Bz * Bz)) * (2.0 * Bz * pp) + gbdrift;
        /* radial drift ~ (b x grad B) . grad x with grad x ~ dq/dpsi grad psi: q_rho |grad rho| */
        double gbdrift0 = psi_r * F / Rz * q_rho * grz * dB2 / (Bz * Bz * Bz * Bz);

        g->bmag[iz]    = Bz;
        g->gradpar[iz] = gradpar;
        /* dB/dz = (dB^2/dl) / (2B) * dl/dz, with dl/dz = (Bp/B) / gradpar */
        g->dbdz[iz]    = dB2 / (2.0 * Bz) * (Bpz / Bz) / gradpar;
        g->gyy[iz]     = gds2;
        g->gxy[iz]     = -gds21 / shat;
        g->gxx[iz]     = gds22 / (shat * shat);
        g->cv_y[iz]    = -0.5 * cvdrift;
        g->gb_y[iz]    = -0.5 * gbdrift;
        g->cv_x[iz]    = 0.5 * gbdrift0 / shat;
        g->gb_x[iz]    = 0.5 * gbdrift0 / shat;
        g->grho[iz]    = grz;
        g->jac[iz]     = 1.0 / (Bz * gradpar);
    }

    free(R); free(J); free(ell); free(gradr); free(sinu); free(Rc); free(Bp); free(B);
    free(nu); free(I1); free(I2); free(I3); free(D); free(Lpar); free(th_ea); free(dB2dl);
    free(dBdr);
    return 0;
}

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
    g->grho     = tgk_malloc_aligned(sizeof(double) * nz);
    g->jac      = tgk_malloc_aligned(sizeof(double) * nz);
    g->z_weight = tgk_malloc_aligned(sizeof(double) * nz);

    double z_max = M_PI * p->z_Period;
    g->dz = (2.0 * z_max) / (double)nz;
    g->kxfac = 1.0;

    /* Cell-centered uniform parallel grid in [-pi*N, pi*N) */
    for (int iz = 0; iz < nz; iz++) {
        g->z[iz] = -z_max + ((double)iz + 0.5) * g->dz;
    }

    if (p->geom_type == TGK_GEOM_MILLER) {
        if (miller_init(p, g) != 0) return -1;
    } else {
        double q0    = p->q0;
        double shat  = p->shat;
        double eps   = p->eps;
        double rmaj  = p->rmaj;
        double alpha = p->alpha_mhd;

        for (int iz = 0; iz < nz; iz++) {
            /* Standard analytic s-alpha local equilibrium model (Dimits et al. 2000) */
            double z = g->z[iz];
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

            /* omega_d_y = -(cos(z) + (shat*z - alpha*sin(z))*sin(z)) / R0, curvature = grad-B */
            double d_y = -(cosz + lambda * sinz) / rmaj;
            double d_x = -(sinz) / rmaj;

            g->cv_y[iz] = d_y;
            g->cv_x[iz] = d_x;
            g->gb_y[iz] = d_y;
            g->gb_x[iz] = d_x;
            g->grho[iz] = 1.0;

            g->jac[iz] = 1.0 / (b * g->gradpar[iz]);
        }
    }

    /* Normalized flux-surface weights and <|grad rho|> for flux normalization */
    double w_sum = 0.0;
    for (int iz = 0; iz < nz; iz++) w_sum += g->jac[iz];
    g->grho_avg = 0.0;
    for (int iz = 0; iz < nz; iz++) {
        g->z_weight[iz] = g->jac[iz] / w_sum;
        g->grho_avg += g->z_weight[iz] * g->grho[iz];
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
    tgk_free_aligned(g->grho);
    tgk_free_aligned(g->jac);
    tgk_free_aligned(g->z_weight);
    memset(g, 0, sizeof(*g));
}
