#ifndef TGK_H
#define TGK_H

#include "simd.h"
#include <pthread.h>
#include <stdio.h>
#include <stdbool.h>

#define TGK_MAX_SPECIES 4
#define TGK_MAX_THREADS 64

typedef enum {
    TGK_GEOM_S_ALPHA = 0,
    TGK_GEOM_MILLER  = 1
} tgk_geom_type;

typedef struct {
    char name[32];
    double charge;      /* Z_s / e */
    double mass;        /* m_s / m_ref */
    double dens;        /* n_s / n_ref */
    double temp;        /* T_s / T_ref */
    double fprim;       /* R / L_n */
    double tprim;       /* R / L_T */
    double nu;          /* collision frequency nu_ss */
} tgk_species_param;

typedef struct {
    /* Grid dimensions (after dealiasing) */
    int nkx;
    int nky;
    int nz;
    int nv;
    int nmu;
    int nspecies;

    /* Physical domain */
    double kx_max;      /* or box size lx */
    double ky_min;      /* minimum non-zero ky */
    double z_Period;    /* number of 2*pi poloidal turns, default 1.0 */
    double vmax;        /* v_parallel max in thermal units, default 3.0 */

    /* Geometry parameters */
    tgk_geom_type geom_type;
    double q0;          /* safety factor q */
    double shat;        /* magnetic shear s_hat = (r/q) dq/dr */
    double eps;         /* r / R0 inverse aspect ratio */
    double rmaj;        /* R0 / a major radius */
    double alpha_mhd;   /* MHD alpha parameter for s-alpha */
    /* Miller geometry parameters */
    double kappa;       /* elongation */
    double s_kappa;     /* r/kappa dkappa/dr */
    double delta;       /* triangularity */
    double s_delta;     /* r ddelta/dr */
    double shift;       /* Shafranov shift dR0/dr */

    /* Physics flags & parameters */
    int adiabatic_electrons; /* 1 if electrons are adiabatic, 0 if kinetic */
    double tau_e;       /* Z_eff * T_i / T_e for adiabatic electrons (default 1.0) */
    double beta;        /* reference beta = 8*pi*n_ref*T_ref / B_ref^2 (0 for electrostatic) */
    int nonlinear;      /* 1 to include ExB nonlinearity, 0 for linear run */
    int imex;           /* 1 to enable implicit electron streaming + collisions */
    double hyper_z;     /* fourth-order parallel hyperdiffusion coefficient */

    /* Time integration & run control */
    int nsteps;
    double dt_max;
    double cfl_frac;
    double t_max;
    int nprint;         /* steps between trace output */
    int nsnap;          /* steps between binary snapshots (0 = disable) */
    int ncheck;         /* steps between checkpoints (0 = disable) */
    int nthreads;       /* number of pthreads (0 = auto P-cores) */
    int seed;           /* random/initial perturbation seed */
    double init_amp;    /* initial perturbation amplitude */
    int init_zonal_only;/* 1 for Rosenbluth-Hinton zonal flow test */

    /* Species array */
    tgk_species_param spec[TGK_MAX_SPECIES];

    /* Output prefix */
    char out_prefix[256];
    char restart_file[256];
} tgk_param;

/* Geometry arrays along z [nz] */
typedef struct {
    double *z;          /* poloidal angle z in [-pi*N, pi*N) */
    double dz;
    double *bmag;       /* B(z) / B_ref */
    double *dbdz;       /* dB/dz (for mirror force) */
    double *gradpar;    /* b_hat . grad(z) parallel derivative metric */
    double *gxx;        /* |grad x|^2 */
    double *gxy;        /* grad x . grad y */
    double *gyy;        /* |grad y|^2 */
    double *cv_x;       /* curvature + grad-B drift x component (multiplykx) */
    double *cv_y;       /* curvature + grad-B drift y component (multiply ky) */
    double *gb_x;       /* pure grad-B drift x component */
    double *gb_y;       /* pure grad-B drift y component */
    double *jac;        /* flux-tube Jacobian J(z) */
    double *z_weight;   /* normalized flux-surface average weights w_z, sum(w_z) = 1 */
} tgk_geom;

/* Grids, quadrature, precomputed Bessel functions, and twist-and-shift connections */
typedef struct {
    int nkx, nky, nz, nv, nmu, nspecies;
    size_t nplane;      /* nky * nkx */
    size_t nfield;      /* nz * nky * nkx */
    size_t nslab;       /* nz * nky * nkx (size of one (s, mu, v) slab) */
    size_t ntot;        /* nspecies * nmu * nv * nz * nky * nkx */

    double *kx;         /* [nkx] in FFT order: 0, dkx, ..., -dkx */
    double *ky;         /* [nky] non-negative: 0, dky, ..., (nky-1)*dky */
    double *vpar;       /* [nv] uniform grid in [-vmax, vmax] */
    double dv;
    double *mu;         /* [nmu] Gauss-Laguerre nodes */
    double *w_mu;       /* [nmu] Gauss-Laguerre weights (including exp(mu) factor removed) */
    double *w_v;        /* [nv] velocity integration weights (trapezoidal / Simpson) */

    /* Equilibrium Maxwellian F0(s, mu, v, z) = (2*pi)^(-1/2) * exp(-v^2/2 - mu*B(z))
     * Note: Gauss-Laguerre integrates int_0^inf exp(-x) f(x) dx, where x = mu * B_ref.
     * We store combined velocity-space integration weights w_vel[s, mu, v, z] and F0[s, mu, v, z]. */
    tgk_real *f0;       /* [nspecies][nmu][nv][nz] equilibrium Maxwellian factor */
    double   *w_vel;    /* [nspecies][nmu][nv][nz] quadrature weight d^3v for field integral */

    /* Bessel function J0(k_perp * rho_s) and Gamma0(b_s) */
    tgk_real *j0;       /* [nspecies][nmu][nz][nky][nkx] */
    double   *qn_denom; /* [nz][nky][nkx] quasineutrality denominator */
    double   *zf_fac;   /* [nkx] zonal-flow flux-surface-averaged correction factor for ky=0 */
    double   *amp_denom;/* [nz][nky][nkx] Ampere's law denominator */
    double   *kperp2;   /* [nz][nky][nkx] k_perp^2(kx, ky, z) */

    /* Twist-and-shift boundary mapping at z boundaries:
     * For each (ky, kx), what kx index does it connect to at z = -pi (+1 shift) and z = +pi (-1 shift)?
     * Wait: standard twist-and-shift: kx' = kx + 2*pi * shat * ky.
     * If index is out of range, connects to open boundary (0 amplitude). */
    int *ts_minus;      /* [nky * nkx] index of connected kx at left boundary (z < 0), or -1 */
    int *ts_plus;       /* [nky * nkx] index of connected kx at right boundary (z >= nz), or -1 */
} tgk_grid;

/* Specialized 2D real-to-complex / complex-to-real FFT plan with folded 2/3-rule dealiasing */
typedef struct {
    int nkx, nky;       /* spectral dimensions (dealiased) */
    int nx_pad, ny_pad; /* padded physical dimensions (>= 3/2 * nkx, 3/2 * nky, radix 2,3,5) */
    int ny_r2c;         /* ny_pad / 2 + 1 */
    double norm;        /* 1.0 / (nx_pad * ny_pad) */

    /* Precomputed twiddle tables for 1D FFTs of length nx_pad and ny_pad */
    tgk_complex *tw_x_fwd;
    tgk_complex *tw_x_inv;
    tgk_complex *tw_y_fwd;
    tgk_complex *tw_y_inv;
    tgk_complex *tw_r2c;
} tgk_fft_plan;

/* Per-thread scratch workspace for zero-allocation RHS and field reduction */
typedef struct {
    /* Thread-local real-space fields dchi/dx and dchi/dy for current (s, mu, v, z) */
    tgk_real *rx_dchi_dx;   /* [nz * nx_pad * ny_pad] or [nx_pad * ny_pad] */
    tgk_real *rx_dchi_dy;
    tgk_real *rx_dh_dx;
    tgk_real *rx_dh_dy;
    tgk_real *rx_nl;
    tgk_complex *cx_buf1;
    tgk_complex *cx_buf2;

    /* Thread-local FP64 field integral accumulators [nz * nky * nkx] */
    tgk_complex64 *qn_num_loc;
    tgk_complex64 *amp_num_loc;

    /* CFL max rates local to thread */
    double vmax_nl;
    /* Cached (s, mu) index in thread workspace for reusing d(J0*phi)/dx, dy across all nv slabs */
    int cached_sm;
    tgk_real *rx_phi_dx_z;  /* [nz * nx_pad * ny_pad] */
    tgk_real *rx_phi_dy_z;  /* [nz * nx_pad * ny_pad] */
    tgk_real *rx_apar_dx_z; /* [nz * nx_pad * ny_pad] */
    tgk_real *rx_apar_dy_z; /* [nz * nx_pad * ny_pad] */
} tgk_thread_ws;

/* Implicit electron streaming & collision solver structures */
typedef struct {
    double dt_cached;
    /* Pre-factorized banded LU matrices or Thomas-algorithm coefficients */
    int initialized;
} tgk_imex_solver;

/* Forward declaration of main simulation context */
typedef struct tgk_ctx tgk_ctx;

typedef struct {
    pthread_t thread;
    int tid;
    tgk_ctx *ctx;
} tgk_worker;

struct tgk_ctx {
    tgk_param param;
    tgk_geom  geom;
    tgk_grid  grid;
    tgk_fft_plan fft;
    tgk_imex_solver imex;

    /* 5D distribution function arrays [nspecies][nmu][nv][nz][nky][nkx] */
    tgk_complex *h;         /* current state h^n */
    tgk_complex *h_stage;   /* intermediate stage state h^(s) */
    tgk_complex *h_rhs;     /* RK weighted RHS accumulator */

    /* 3D electromagnetic fields [nz][nky][nkx] */
    tgk_complex *phi;       /* electrostatic potential */
    tgk_complex *apar;      /* parallel vector potential A_|| */

    /* Global FP64 field reduction accumulators [nz][nky][nkx] */
    tgk_complex64 *qn_num;
    tgk_complex64 *amp_num;

    /* Thread pool and work-stealing atomic counter */
    int nthreads;
    tgk_worker workers[TGK_MAX_THREADS];
    tgk_thread_ws ws[TGK_MAX_THREADS];
    _Atomic int next_slab;
    _Atomic int steal_q[TGK_MAX_THREADS];
    int         steal_end[TGK_MAX_THREADS];

    /* Synchronization barrier for thread pool */
    pthread_mutex_t pool_mtx;
    pthread_cond_t  pool_cv_start;
    pthread_cond_t  pool_cv_done;
    int pool_gen;
    int pool_active;
    int pool_quit;
    void (*pool_fn)(tgk_ctx *ctx, int tid);
    double rk_dt;
    double rk_b;            /* weight in h_accum += rk_b * dt * rhs */
    double rk_c;            /* stage weight for h_stage = h + rk_c * dt * rhs (0 on last stage) */
    int    rk_stage_idx;

    /* Simulation clock and adaptive timestep */
    int step;
    double time;
    double dt;
    double cfl_linear;

    /* Diagnostics history for growth rate calculation */
    double prev_time;
    double *prev_ky_amp;    /* [nky] previous |phi(ky)| for instantaneous growth rate */
    double *prev_ky_phase;  /* [nky] previous phase(phi(ky)) for real frequency */
    double prev_free_energy;
};

/* Function prototypes across the 10 modules */

/* geom.c */
int  tgk_geom_init(const tgk_param *p, tgk_geom *g);
void tgk_geom_free(tgk_geom *g);

/* grid.c */
int  tgk_grid_init(const tgk_param *p, const tgk_geom *g, tgk_grid *gr);
void tgk_grid_free(tgk_grid *gr);
double tgk_bessel_j0(double x);
double tgk_gamma0(double b);

/* fft.c */
int  tgk_fft_init(tgk_fft_plan *plan, int nkx, int nky);
void tgk_fft_free(tgk_fft_plan *plan);
void tgk_fft_c2r_dealiased(const tgk_fft_plan *plan, const tgk_complex *spec_in,
                           tgk_real *real_out, tgk_complex *scratch1, tgk_complex *scratch2);
void tgk_fft_c2r_pair_dealiased(const tgk_fft_plan *plan, const tgk_complex *spec_a,
                                const tgk_complex *spec_b, tgk_real *real_a, tgk_real *real_b,
                                tgk_complex *scratch1, tgk_complex *scratch2);
void tgk_fft_c2r_grad_dealiased(const tgk_fft_plan *plan, const tgk_complex *spec_f,
                                const double *kx, const double *ky,
                                tgk_real *real_df_dx, tgk_real *real_df_dy,
                                tgk_complex *scratch1, tgk_complex *scratch2);
void tgk_fft_r2c_dealiased(const tgk_fft_plan *plan, const tgk_real *real_in,
                           tgk_complex *spec_out, tgk_complex *scratch1, tgk_complex *scratch2);
void tgk_fft_r2c_pair_dealiased(const tgk_fft_plan *plan, const tgk_real *real_a,
                                const tgk_real *real_b, tgk_complex *spec_a, tgk_complex *spec_b,
                                tgk_complex *scratch1, tgk_complex *scratch2);
void tgk_fft_1d_c2c(tgk_complex *in, tgk_complex *out, int n, int inv);

/* field.c */
void tgk_field_clear_accum(tgk_ctx *ctx);
void tgk_field_reduce_and_solve(tgk_ctx *ctx);
void tgk_field_compute_from_h(tgk_ctx *ctx, const tgk_complex *h_in);

/* rhs.c */
void tgk_rhs_eval_fused_stage(tgk_ctx *ctx, int tid);
double tgk_rhs_compute_linear_cfl(const tgk_ctx *ctx);

/* coll.c */
void tgk_coll_step_all(tgk_ctx *ctx, tgk_complex *h_inout, double dt);

/* imex.c */
int  tgk_imex_init(tgk_ctx *ctx);
void tgk_imex_free(tgk_ctx *ctx);
void tgk_imex_solve_electrons(tgk_ctx *ctx, tgk_complex *h_inout, double dt);

/* Fetch a plane element at (iz_target, iky, ikx) applying twist-and-shift boundary condition
 * when iz_target crosses the ends of the flux tube [0, nz-1]. */
static inline tgk_complex tgk_fetch_z_ts(const tgk_grid *gr, const tgk_complex *slab_base,
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

/* diag.c */
typedef struct {
    double heat_flux_i;
    double heat_flux_e;
    double part_flux_i;
    double part_flux_e;
    double chi_i;
    double chi_e;
    double phi2_total;
    double phi2_zonal;
    double free_energy;
    double fe_drive;
    double fe_dissip;
    double fe_coll;
    double fe_residual;
    double gamma_max;
    double omega_max;
} tgk_diag_out;

void tgk_diag_compute(tgk_ctx *ctx, const tgk_complex *h_in, tgk_diag_out *out);
void tgk_diag_write_trace(tgk_ctx *ctx, FILE *fp, const tgk_diag_out *d);
int  tgk_diag_write_snapshot(const tgk_ctx *ctx, const char *filename);
int  tgk_checkpoint_write(const tgk_ctx *ctx, const char *filename);
int  tgk_checkpoint_read(tgk_ctx *ctx, const char *filename);

/* main.c / context management */
int  tgk_param_defaults(tgk_param *p);
int  tgk_param_parse_file(tgk_param *p, const char *filename);
int  tgk_ctx_init(tgk_ctx *ctx, const tgk_param *p);
void tgk_ctx_free(tgk_ctx *ctx);
void tgk_init_perturbation(tgk_ctx *ctx);
void tgk_step_rk4(tgk_ctx *ctx);
void tgk_pool_run(tgk_ctx *ctx, void (*fn)(tgk_ctx *ctx, int tid));

/* Aligned memory allocation helper */
void *tgk_malloc_aligned(size_t size);
void  tgk_free_aligned(void *ptr);

#endif /* TGK_H */
