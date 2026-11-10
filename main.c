#include "tgk.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdatomic.h>
#include <time.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <pthread/qos.h>
#endif

void *tgk_malloc_aligned(size_t size) {
    if (size == 0) size = TGK_CACHELINE;
    size_t aligned_sz = (size + TGK_CACHELINE - 1) & ~((size_t)TGK_CACHELINE - 1);
    void *ptr = NULL;
    if (posix_memalign(&ptr, TGK_CACHELINE, aligned_sz) != 0 || !ptr) {
        fprintf(stderr, "tgk: fatal out of memory allocating %zu bytes\n", aligned_sz);
        abort();
    }
    memset(ptr, 0, aligned_sz);
    return ptr;
}

void tgk_free_aligned(void *ptr) {
    if (ptr) free(ptr);
}

static int detect_apple_p_cores(void) {
#ifdef __APPLE__
    int pcores = 0;
    size_t len = sizeof(pcores);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &pcores, &len, NULL, 0) == 0 && pcores > 0) {
        return pcores;
    }
    if (sysctlbyname("hw.physicalcpu", &pcores, &len, NULL, 0) == 0 && pcores > 0) {
        return pcores;
    }
#endif
    return 8;
}

int tgk_param_defaults(tgk_param *p) {
    memset(p, 0, sizeof(*p));
    /* Default Cyclone Base Case parameters (Dimits et al. PoP 2000) */
    p->nkx                 = 32;
    p->nky                 = 16;
    p->nz                  = 24;
    p->nv                  = 32;
    p->nmu                 = 8;
    p->nspecies            = 1;

    p->kx_max              = 0.0; /* auto from twist-and-shift */
    p->ky_min              = 0.05;
    p->z_Period            = 1.0;
    p->vmax                = 3.5;

    p->geom_type           = TGK_GEOM_S_ALPHA;
    p->q0                  = 1.4;
    p->shat                = 0.796;
    p->eps                 = 0.18;
    p->rmaj                = 2.77778; /* R0/a so R/L_T = 6.92, R/L_n = 2.22 */
    p->alpha_mhd           = 0.0;

    p->kappa               = 1.0;
    p->s_kappa             = 0.0;
    p->delta               = 0.0;
    p->s_delta             = 0.0;
    p->shift               = 0.0;
    p->betaprim            = 0.0;

    p->adiabatic_electrons = 1;
    p->tau_e               = 1.0;
    p->beta                = 0.0;
    p->nonlinear           = 1;
    p->imex                = 0;
    p->hyper_z             = 0.15;

    p->nsteps              = 500;
    p->dt_max              = 0.05;
    p->cfl_frac            = 0.85;
    p->t_max               = 1000.0;
    p->nprint              = 10;
    p->nsnap               = 0;
    p->ncheck              = 0;
    p->nthreads            = 0; /* auto P-cores */
    p->seed                = 1337;
    p->init_amp            = 1e-3;
    p->init_zonal_only     = 0;

    /* Default species 0: Deuterium ions */
    snprintf(p->spec[0].name, sizeof(p->spec[0].name), "ions");
    p->spec[0].charge      = 1.0;
    p->spec[0].mass        = 1.0;
    p->spec[0].dens        = 1.0;
    p->spec[0].temp        = 1.0;
    p->spec[0].fprim       = 2.22;
    p->spec[0].tprim       = 6.92;
    p->spec[0].nu          = 0.0;

    /* Default species 1: Kinetic electrons (if nspecies >= 2) */
    snprintf(p->spec[1].name, sizeof(p->spec[1].name), "electrons");
    p->spec[1].charge      = -1.0;
    p->spec[1].mass        = 0.00027244; /* m_e / m_D = 1 / 3670 */
    p->spec[1].dens        = 1.0;
    p->spec[1].temp        = 1.0;
    p->spec[1].fprim       = 2.22;
    p->spec[1].tprim       = 6.92;
    p->spec[1].nu          = 0.0;

    snprintf(p->out_prefix, sizeof(p->out_prefix), "tgk_run");
    p->restart_file[0]     = '\0';
    return 0;
}

static char *trim_ws(char *s) {
    while (*s && isspace((unsigned char)*s)) s++;
    if (*s == '\0') return s;
    char *end = s + strlen(s) - 1;
    while (end > s && isspace((unsigned char)*end)) *end-- = '\0';
    return s;
}

/* Parse key = value input file. Per Section 7: Unknown keys are errors, not warnings! */
int tgk_param_parse_file(tgk_param *p, const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "tgk: error opening input file '%s'\n", filename);
        return -1;
    }

    char line[512];
    int lineno = 0;
    while (fgets(line, sizeof(line), fp)) {
        lineno++;
        /* Strip comments starting with # or ; */
        char *comment = strpbrk(line, "#;");
        if (comment) *comment = '\0';

        char *trimmed = trim_ws(line);
        if (*trimmed == '\0') continue;

        char *eq = strchr(trimmed, '=');
        if (!eq) {
            fprintf(stderr, "tgk: syntax error on line %d of '%s': missing '='\n", lineno, filename);
            fclose(fp);
            return -1;
        }
        *eq = '\0';
        char *key = trim_ws(trimmed);
        char *val = trim_ws(eq + 1);

        if      (strcmp(key, "nkx") == 0)                 p->nkx = atoi(val);
        else if (strcmp(key, "nky") == 0)                 p->nky = atoi(val);
        else if (strcmp(key, "nz") == 0)                  p->nz = atoi(val);
        else if (strcmp(key, "nv") == 0)                  p->nv = atoi(val);
        else if (strcmp(key, "nmu") == 0)                 p->nmu = atoi(val);
        else if (strcmp(key, "nspecies") == 0)            p->nspecies = atoi(val);
        else if (strcmp(key, "kx_max") == 0)              p->kx_max = atof(val);
        else if (strcmp(key, "ky_min") == 0)              p->ky_min = atof(val);
        else if (strcmp(key, "z_period") == 0)            p->z_Period = atof(val);
        else if (strcmp(key, "vmax") == 0)                p->vmax = atof(val);
        else if (strcmp(key, "geometry") == 0) {
            if (strcmp(val, "s-alpha") == 0 || strcmp(val, "s_alpha") == 0) p->geom_type = TGK_GEOM_S_ALPHA;
            else if (strcmp(val, "miller") == 0) p->geom_type = TGK_GEOM_MILLER;
            else {
                fprintf(stderr, "tgk: unknown geometry '%s' on line %d\n", val, lineno);
                fclose(fp);
                return -1;
            }
        }
        else if (strcmp(key, "q0") == 0)                  p->q0 = atof(val);
        else if (strcmp(key, "shat") == 0)                p->shat = atof(val);
        else if (strcmp(key, "eps") == 0)                 p->eps = atof(val);
        else if (strcmp(key, "rmaj") == 0)                p->rmaj = atof(val);
        else if (strcmp(key, "alpha_mhd") == 0)           p->alpha_mhd = atof(val);
        else if (strcmp(key, "kappa") == 0)               p->kappa = atof(val);
        else if (strcmp(key, "s_kappa") == 0)             p->s_kappa = atof(val);
        else if (strcmp(key, "delta") == 0)               p->delta = atof(val);
        else if (strcmp(key, "s_delta") == 0)             p->s_delta = atof(val);
        else if (strcmp(key, "shift") == 0)               p->shift = atof(val);
        else if (strcmp(key, "betaprim") == 0)            p->betaprim = atof(val);
        else if (strcmp(key, "adiabatic_electrons") == 0) p->adiabatic_electrons = atoi(val);
        else if (strcmp(key, "tau_e") == 0)               p->tau_e = atof(val);
        else if (strcmp(key, "beta") == 0)                p->beta = atof(val);
        else if (strcmp(key, "nonlinear") == 0)           p->nonlinear = atoi(val);
        else if (strcmp(key, "imex") == 0)                p->imex = atoi(val);
        else if (strcmp(key, "hyper_z") == 0)             p->hyper_z = atof(val);
        else if (strcmp(key, "nsteps") == 0)              p->nsteps = atoi(val);
        else if (strcmp(key, "dt_max") == 0)              p->dt_max = atof(val);
        else if (strcmp(key, "cfl_frac") == 0)            p->cfl_frac = atof(val);
        else if (strcmp(key, "t_max") == 0)               p->t_max = atof(val);
        else if (strcmp(key, "nprint") == 0)              p->nprint = atoi(val);
        else if (strcmp(key, "nsnap") == 0)               p->nsnap = atoi(val);
        else if (strcmp(key, "ncheck") == 0)              p->ncheck = atoi(val);
        else if (strcmp(key, "nthreads") == 0)            p->nthreads = atoi(val);
        else if (strcmp(key, "seed") == 0)                p->seed = atoi(val);
        else if (strcmp(key, "init_amp") == 0)            p->init_amp = atof(val);
        else if (strcmp(key, "init_zonal_only") == 0)     p->init_zonal_only = atoi(val);
        else if (strcmp(key, "out_prefix") == 0)          snprintf(p->out_prefix, sizeof(p->out_prefix), "%s", val);
        else if (strcmp(key, "restart_file") == 0)        snprintf(p->restart_file, sizeof(p->restart_file), "%s", val);
        else if (strncmp(key, "spec", 4) == 0 && isdigit((unsigned char)key[4]) && key[5] == '.') {
            int s = key[4] - '0';
            if (s < 0 || s >= TGK_MAX_SPECIES) {
                fprintf(stderr, "tgk: species index %d out of range on line %d\n", s, lineno);
                fclose(fp);
                return -1;
            }
            const char *sub = key + 6;
            if      (strcmp(sub, "name") == 0)   snprintf(p->spec[s].name, sizeof(p->spec[s].name), "%s", val);
            else if (strcmp(sub, "charge") == 0) p->spec[s].charge = atof(val);
            else if (strcmp(sub, "mass") == 0)   p->spec[s].mass   = atof(val);
            else if (strcmp(sub, "dens") == 0)   p->spec[s].dens   = atof(val);
            else if (strcmp(sub, "temp") == 0)   p->spec[s].temp   = atof(val);
            else if (strcmp(sub, "fprim") == 0)  p->spec[s].fprim  = atof(val);
            else if (strcmp(sub, "tprim") == 0)  p->spec[s].tprim  = atof(val);
            else if (strcmp(sub, "nu") == 0)     p->spec[s].nu     = atof(val);
            else {
                fprintf(stderr, "tgk: unknown species key '%s' on line %d\n", key, lineno);
                fclose(fp);
                return -1;
            }
        } else {
            fprintf(stderr, "tgk: unknown configuration key '%s' on line %d of '%s'\n", key, lineno, filename);
            fclose(fp);
            return -1;
        }
    }

    fclose(fp);
    return 0;
}

/* Worker thread entry point with Apple Silicon QoS USER_INTERACTIVE for P-core affinity */
static void *tgk_worker_main(void *arg) {
    tgk_worker *w = (tgk_worker *)arg;
    tgk_ctx *ctx  = w->ctx;

#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

    int my_gen = 0;
    while (1) {
        pthread_mutex_lock(&ctx->pool_mtx);
        while (ctx->pool_gen == my_gen && !ctx->pool_quit) {
            pthread_cond_wait(&ctx->pool_cv_start, &ctx->pool_mtx);
        }
        if (ctx->pool_quit) {
            pthread_mutex_unlock(&ctx->pool_mtx);
            break;
        }
        my_gen = ctx->pool_gen;
        void (*fn)(tgk_ctx *, int) = ctx->pool_fn;
        pthread_mutex_unlock(&ctx->pool_mtx);

        if (fn) {
            fn(ctx, w->tid);
        }

        pthread_mutex_lock(&ctx->pool_mtx);
        ctx->pool_active--;
        if (ctx->pool_active == 0) {
            pthread_cond_signal(&ctx->pool_cv_done);
        }
        pthread_mutex_unlock(&ctx->pool_mtx);
    }
    return NULL;
}

void tgk_pool_run(tgk_ctx *ctx, void (*fn)(tgk_ctx *ctx, int tid)) {
    int total_slabs = ctx->grid.nspecies * ctx->grid.nmu * ctx->grid.nv;
    int nt = ctx->nthreads;
    int chunk = (total_slabs + nt - 1) / nt;
    for (int t = 0; t < nt; t++) {
        int start = t * chunk;
        int end   = start + chunk;
        if (start > total_slabs) start = total_slabs;
        if (end   > total_slabs) end   = total_slabs;
        atomic_store_explicit(&ctx->steal_q[t], start, memory_order_relaxed);
        ctx->steal_end[t] = end;
    }
    atomic_store_explicit(&ctx->next_slab, 0, memory_order_relaxed);
    if (ctx->nthreads <= 1) {
        fn(ctx, 0);
        return;
    }

    pthread_mutex_lock(&ctx->pool_mtx);
    ctx->pool_fn     = fn;
    ctx->pool_active = ctx->nthreads - 1;
    ctx->pool_gen++;
    pthread_cond_broadcast(&ctx->pool_cv_start);
    pthread_mutex_unlock(&ctx->pool_mtx);

    /* Main thread participates as worker tid = 0 */
    fn(ctx, 0);

    pthread_mutex_lock(&ctx->pool_mtx);
    while (ctx->pool_active > 0) {
        pthread_cond_wait(&ctx->pool_cv_done, &ctx->pool_mtx);
    }
    pthread_mutex_unlock(&ctx->pool_mtx);
}

int tgk_ctx_init(tgk_ctx *ctx, const tgk_param *p) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->param = *p;

    ctx->nthreads = (p->nthreads > 0) ? p->nthreads : detect_apple_p_cores();
    if (ctx->nthreads > TGK_MAX_THREADS) ctx->nthreads = TGK_MAX_THREADS;

    if (tgk_geom_init(&ctx->param, &ctx->geom) != 0) return -1;
    if (tgk_grid_init(&ctx->param, &ctx->geom, &ctx->grid) != 0) return -1;
    if (tgk_fft_init(&ctx->fft, ctx->grid.nkx, ctx->grid.nky) != 0) return -1;
    if (tgk_imex_init(ctx) != 0) return -1;

    size_t ntot   = ctx->grid.ntot;
    size_t nfield = ctx->grid.nfield;

    /* Allocate the full-size 5D buffers (h, two stage buffers, h_rhs) */
    ctx->h       = tgk_malloc_aligned(sizeof(tgk_complex) * ntot);
    ctx->h_stage = tgk_malloc_aligned(sizeof(tgk_complex) * ntot);
    ctx->h_stage2 = tgk_malloc_aligned(sizeof(tgk_complex) * ntot);
    ctx->h_rhs   = tgk_malloc_aligned(sizeof(tgk_complex) * ntot);

    /* Allocate 3D fields and global FP64 accumulators */
    ctx->phi     = tgk_malloc_aligned(sizeof(tgk_complex) * nfield);
    ctx->apar    = tgk_malloc_aligned(sizeof(tgk_complex) * nfield);
    ctx->qn_num  = tgk_malloc_aligned(sizeof(tgk_complex64) * nfield);
    ctx->amp_num = tgk_malloc_aligned(sizeof(tgk_complex64) * nfield);

    ctx->prev_ky_amp   = tgk_malloc_aligned(sizeof(double) * ctx->grid.nky);
    ctx->prev_ky_phase = tgk_malloc_aligned(sizeof(double) * ctx->grid.nky);

    /* Allocate per-thread scratch workspaces */
    size_t n_real  = (size_t)ctx->fft.nx_pad * (size_t)ctx->fft.ny_pad;
    size_t n_realz = (size_t)ctx->grid.nz * n_real;
    size_t n_cx1   = 2 * ctx->grid.nplane + 2 * (size_t)ctx->fft.nx_pad * (size_t)ctx->fft.ny_pad;
    size_t n_cx2   = 4 * (size_t)(ctx->fft.nx_pad + ctx->fft.ny_pad);

    for (int t = 0; t < ctx->nthreads; t++) {
        ctx->ws[t].rx_dchi_dx   = tgk_malloc_aligned(sizeof(tgk_real) * n_real);
        ctx->ws[t].rx_dchi_dy   = tgk_malloc_aligned(sizeof(tgk_real) * n_real);
        ctx->ws[t].rx_dh_dx     = tgk_malloc_aligned(sizeof(tgk_real) * n_real);
        ctx->ws[t].rx_dh_dy     = tgk_malloc_aligned(sizeof(tgk_real) * n_real);
        ctx->ws[t].rx_nl        = tgk_malloc_aligned(sizeof(tgk_real) * n_real);
        ctx->ws[t].rx_phi_dx_z  = tgk_malloc_aligned(sizeof(tgk_real) * n_realz);
        ctx->ws[t].rx_phi_dy_z  = tgk_malloc_aligned(sizeof(tgk_real) * n_realz);
        ctx->ws[t].rx_apar_dx_z = tgk_malloc_aligned(sizeof(tgk_real) * n_realz);
        ctx->ws[t].rx_apar_dy_z = tgk_malloc_aligned(sizeof(tgk_real) * n_realz);
        ctx->ws[t].cx_buf1      = tgk_malloc_aligned(sizeof(tgk_complex) * n_cx1);
        ctx->ws[t].cx_buf2      = tgk_malloc_aligned(sizeof(tgk_complex) * n_cx2);
        ctx->ws[t].qn_num_loc   = tgk_malloc_aligned(sizeof(tgk_complex64) * nfield);
        ctx->ws[t].amp_num_loc  = tgk_malloc_aligned(sizeof(tgk_complex64) * nfield);
        ctx->ws[t].cached_sm    = -1;
    }

    pthread_mutex_init(&ctx->pool_mtx, NULL);
    pthread_cond_init(&ctx->pool_cv_start, NULL);
    pthread_cond_init(&ctx->pool_cv_done, NULL);

#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

    for (int t = 1; t < ctx->nthreads; t++) {
        ctx->workers[t].tid = t;
        ctx->workers[t].ctx = ctx;
        pthread_create(&ctx->workers[t].thread, NULL, tgk_worker_main, &ctx->workers[t]);
    }

    ctx->cfl_linear = tgk_rhs_compute_linear_cfl(ctx);
    ctx->dt = fmin(ctx->param.dt_max, ctx->param.cfl_frac / ctx->cfl_linear);

    return 0;
}

void tgk_ctx_free(tgk_ctx *ctx) {
    if (ctx->nthreads > 1) {
        pthread_mutex_lock(&ctx->pool_mtx);
        ctx->pool_quit = 1;
        pthread_cond_broadcast(&ctx->pool_cv_start);
        pthread_mutex_unlock(&ctx->pool_mtx);
        for (int t = 1; t < ctx->nthreads; t++) {
            pthread_join(ctx->workers[t].thread, NULL);
        }
    }
    pthread_mutex_destroy(&ctx->pool_mtx);
    pthread_cond_destroy(&ctx->pool_cv_start);
    pthread_cond_destroy(&ctx->pool_cv_done);

    for (int t = 0; t < ctx->nthreads; t++) {
        tgk_free_aligned(ctx->ws[t].rx_dchi_dx);
        tgk_free_aligned(ctx->ws[t].rx_dchi_dy);
        tgk_free_aligned(ctx->ws[t].rx_dh_dx);
        tgk_free_aligned(ctx->ws[t].rx_dh_dy);
        tgk_free_aligned(ctx->ws[t].rx_nl);
        tgk_free_aligned(ctx->ws[t].rx_phi_dx_z);
        tgk_free_aligned(ctx->ws[t].rx_phi_dy_z);
        tgk_free_aligned(ctx->ws[t].rx_apar_dx_z);
        tgk_free_aligned(ctx->ws[t].rx_apar_dy_z);
        tgk_free_aligned(ctx->ws[t].cx_buf1);
        tgk_free_aligned(ctx->ws[t].cx_buf2);
        tgk_free_aligned(ctx->ws[t].qn_num_loc);
        tgk_free_aligned(ctx->ws[t].amp_num_loc);
    }

    tgk_free_aligned(ctx->h);
    tgk_free_aligned(ctx->h_stage);
    tgk_free_aligned(ctx->h_stage2);
    tgk_free_aligned(ctx->h_rhs);
    tgk_free_aligned(ctx->phi);
    tgk_free_aligned(ctx->apar);
    tgk_free_aligned(ctx->qn_num);
    tgk_free_aligned(ctx->amp_num);
    tgk_free_aligned(ctx->prev_ky_amp);
    tgk_free_aligned(ctx->prev_ky_phase);

    tgk_imex_free(ctx);
    tgk_fft_free(&ctx->fft);
    tgk_grid_free(&ctx->grid);
    tgk_geom_free(&ctx->geom);
}

void tgk_init_perturbation(tgk_ctx *ctx) {
    const tgk_grid  *gr = &ctx->grid;
    const tgk_geom  *gm = &ctx->geom;
    const tgk_param *p  = &ctx->param;

    int nkx = gr->nkx, nky = gr->nky, nz = gr->nz, nv = gr->nv, nmu = gr->nmu;
    size_t nfield = gr->nfield;
    memset(ctx->h, 0, sizeof(tgk_complex) * gr->ntot);

    uint32_t rng = (uint32_t)(p->seed ? p->seed : 1337);

    for (int s = 0; s < gr->nspecies; s++) {
        for (int imu = 0; imu < nmu; imu++) {
            for (int iv = 0; iv < nv; iv++) {
                size_t slab_off = (((size_t)s * nmu + imu) * nv + iv) * nfield;
                for (int iz = 0; iz < nz; iz++) {
                    double z = gm->z[iz];
                    /* Ballooning envelope exp(-(z / 1.5)^2) for clean linear eigenmode startup */
                    double env = p->init_zonal_only ? 1.0 : exp(-0.4 * z * z);
                    size_t vidx = (((size_t)s * nmu + imu) * nv + iv) * nz + iz;
                    double f0 = (double)gr->f0[vidx];

                    for (int iky = 0; iky < nky; iky++) {
                        if (p->init_zonal_only && iky > 0) continue;
                        if (!p->init_zonal_only && iky == 0) continue;

                        for (int ikx = 0; ikx < nkx; ikx++) {
                            if (iky == 0 && ikx == 0) continue;
                            if (p->init_zonal_only && ikx != 1 && ikx != nkx - 1) continue;
                            if (!p->init_zonal_only && p->nonlinear == 0 && ikx != 0) continue;

                            size_t idx = slab_off + ((size_t)iz * nky + iky) * nkx + ikx;
                            if (p->init_zonal_only) {
                                /* Real cosine perturbation in x: F(kx1, 0) = F(-kx1, 0) = init_amp * F0 */
                                ctx->h[idx].re = (tgk_real)(p->init_amp * f0);
                                ctx->h[idx].im = 0.0f;
                            } else if (p->nonlinear == 0) {
                                /* Clean ballooning seed on kx = 0 for linear benchmark */
                                ctx->h[idx].re = (tgk_real)(p->init_amp * env * f0);
                                ctx->h[idx].im = (tgk_real)(0.1 * p->init_amp * sin(z) * env * f0);
                            } else {
                                /* Pseudo-random phases across (kx, ky) for nonlinear turbulence */
                                rng = rng * 1664525U + 1013904223U;
                                double ph = (double)(rng & 0xFFFF) * (2.0 * 3.14159265358979323846 / 65536.0);
                                ctx->h[idx].re = (tgk_real)(p->init_amp * env * f0 * cos(ph));
                                ctx->h[idx].im = (tgk_real)(p->init_amp * env * f0 * sin(ph));
                            }
                        }
                    }
                }
            }
        }
    }

    /* Compute initial fields phi and apar from h^0 */
    tgk_field_compute_from_h(ctx, ctx->h);
}

/* Advance one complete 4th-order Runge-Kutta step with fused stage evaluation & field solve,
 * plus operator-split implicit electron streaming (if enabled) and conserving collisions. */
void tgk_step_rk4(tgk_ctx *ctx) {
    static const double rk_b[4] = { 1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0 };
    static const double rk_c[4] = { 0.5,       0.5,       1.0,       0.0       };

    double dt = ctx->dt;
    ctx->rk_dt = dt;
    double vmax_nl = 0.0;

    for (int stg = 0; stg < 4; stg++) {
        ctx->rk_stage_idx = stg;
        ctx->rk_b         = rk_b[stg];
        ctx->rk_c         = rk_c[stg];

        tgk_field_clear_accum(ctx);
        tgk_pool_run(ctx, tgk_rhs_eval_fused_stage);

        /* Collect nonlinear CFL rate from stage 0 */
        if (stg == 0 && ctx->param.nonlinear) {
            for (int t = 0; t < ctx->nthreads; t++) {
                if (ctx->ws[t].vmax_nl > vmax_nl) vmax_nl = ctx->ws[t].vmax_nl;
            }
        }

        /* Reduce velocity-space integrals accumulated during the fused pass and solve for fields */
        tgk_field_reduce_and_solve(ctx);
    }

    /* Apply implicit electron parallel streaming if IMEX is active */
    if (ctx->param.imex) {
        tgk_imex_solve_electrons(ctx, ctx->h, dt);
    }

    /* Apply momentum- and energy-conserving Dougherty collisions if nu > 0 */
    tgk_coll_step_all(ctx, ctx->h, dt);
    if (ctx->param.imex) {
        tgk_field_compute_from_h(ctx, ctx->h);
    }

    ctx->step++;
    ctx->time += dt;

    /* Adaptive timestep update based on linear + nonlinear CFL limits */
    double total_rate = ctx->cfl_linear + vmax_nl;
    double dt_cfl = (total_rate > 1e-12) ? (ctx->param.cfl_frac * 2.8 / total_rate) : ctx->param.dt_max;
    ctx->dt = fmin(ctx->param.dt_max, dt_cfl);
}

#ifndef TGK_LIBRARY_ONLY
int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s input.txt\n", argv[0]);
        return 1;
    }

    tgk_param param;
    tgk_param_defaults(&param);
    if (tgk_param_parse_file(&param, argv[1]) != 0) {
        return 1;
    }

    /* Known-unvalidated paths (see README, "Validation status") */
    if (!param.adiabatic_electrons && param.imex) {
        fprintf(stderr, "tgk: WARNING: imex = 1 is not validated: it gives spurious growth rates ~3-4x GX's\n"
                        "     for kinetic-electron ITG, even electrostatically. Use imex = 0.\n");
    }
    if (!param.adiabatic_electrons && param.beta > 0.0) {
        fprintf(stderr, "tgk: WARNING: electromagnetic kinetic electrons (beta > 0) are not validated: the\n"
                        "     discrete system has a spurious instability that grows without any gradient drive.\n");
    }

    tgk_ctx ctx;
    if (tgk_ctx_init(&ctx, &param) != 0) {
        fprintf(stderr, "tgk: failed to initialize simulation context\n");
        return 1;
    }

    if (param.restart_file[0] != '\0') {
        if (tgk_checkpoint_read(&ctx, param.restart_file) != 0) {
            fprintf(stderr, "tgk: failed to read checkpoint '%s'\n", param.restart_file);
            tgk_ctx_free(&ctx);
            return 1;
        }
    } else {
        tgk_init_perturbation(&ctx);
    }

    char trace_path[300];
    snprintf(trace_path, sizeof(trace_path), "%s.trace.txt", param.out_prefix);
    FILE *fp_trace = fopen(trace_path, "w");

    printf("TGK (%s precision, %d P-core threads, grid %dx%dx%dx%dx%d x %d species, FFT pad %dx%d)\n",
           TGK_REAL_IS_FP64 ? "FP64" : "FP32-mixed",
           ctx.nthreads,
           ctx.grid.nkx, ctx.grid.nky, ctx.grid.nz, ctx.grid.nv, ctx.grid.nmu, ctx.grid.nspecies,
           ctx.fft.nx_pad, ctx.fft.ny_pad);

    tgk_diag_out diag;
    tgk_diag_compute(&ctx, ctx.h, &diag);
    tgk_diag_write_trace(&ctx, fp_trace, &diag);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    while (ctx.step < param.nsteps && ctx.time < param.t_max) {
        tgk_step_rk4(&ctx);

        if (param.nprint > 0 && (ctx.step % param.nprint == 0 || ctx.step == param.nsteps)) {
            tgk_diag_compute(&ctx, ctx.h, &diag);
            tgk_diag_write_trace(&ctx, fp_trace, &diag);
        }

        if (param.nsnap > 0 && (ctx.step % param.nsnap == 0)) {
            char snap_path[300];
            snprintf(snap_path, sizeof(snap_path), "%s.snap.%06d.bin", param.out_prefix, ctx.step);
            tgk_diag_write_snapshot(&ctx, snap_path);
        }

        if (param.ncheck > 0 && (ctx.step % param.ncheck == 0)) {
            char chk_path[300];
            snprintf(chk_path, sizeof(chk_path), "%s.chk.%06d.bin", param.out_prefix, ctx.step);
            tgk_checkpoint_write(&ctx, chk_path);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
    double ms_per_step  = (ctx.step > 0) ? (1000.0 * elapsed / (double)ctx.step) : 0.0;
    double ms_per_stage = 0.25 * ms_per_step;

    /* Calculate effective memory bandwidth per RK stage (3 reads + 2 writes of h) */
    double bytes_per_stage = 5.0 * (double)ctx.grid.ntot * (double)sizeof(tgk_complex);
    double gbps = (ms_per_stage > 0.0) ? (bytes_per_stage / (ms_per_stage * 1e6)) : 0.0;

    printf("Completed %d steps (t = %.4f) in %.3f s (%.2f ms/step, %.2f ms/stage, %.1f GB/s eff)\n",
           ctx.step, ctx.time, elapsed, ms_per_step, ms_per_stage, gbps);
    printf("Final: |phi|^2 = %.4e, Q_i = %.4e, chi_i = %.4f, gamma = %.4f, omega = %.4f\n",
           diag.phi2_total, diag.heat_flux_i, diag.chi_i, diag.gamma_max, diag.omega_max);

    if (fp_trace) fclose(fp_trace);
    tgk_ctx_free(&ctx);
    return 0;
}
#endif
