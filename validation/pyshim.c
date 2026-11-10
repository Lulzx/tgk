/* ctypes shim: expose TGK's RK4 step as a linear map for Arnoldi eigenvalue analysis. */
#include "../tgk.h"
#include <string.h>
static tgk_ctx g_ctx;
long tgk_py_init(const char *input) {
    tgk_param p; tgk_param_defaults(&p);
    if (tgk_param_parse_file(&p, input)) return -1;
    if (tgk_ctx_init(&g_ctx, &p)) return -2;
    tgk_init_perturbation(&g_ctx);
    return (long)g_ctx.grid.ntot;
}
double tgk_py_dt(void) { return g_ctx.dt; }
void tgk_py_dims(int *d) { d[0]=g_ctx.grid.nspecies; d[1]=g_ctx.grid.nmu; d[2]=g_ctx.grid.nv; d[3]=g_ctx.grid.nz; d[4]=g_ctx.grid.nky; d[5]=g_ctx.grid.nkx; }
void tgk_py_get_h(tgk_complex *out) { memcpy(out, g_ctx.h, sizeof(tgk_complex) * g_ctx.grid.ntot); }
void tgk_py_apply(const tgk_complex *in, tgk_complex *out, int nsteps) {
    memcpy(g_ctx.h, in, sizeof(tgk_complex) * g_ctx.grid.ntot);
    tgk_field_compute_from_h(&g_ctx, g_ctx.h);
    for (int i = 0; i < nsteps; i++) tgk_step_rk4(&g_ctx);
    memcpy(out, g_ctx.h, sizeof(tgk_complex) * g_ctx.grid.ntot);
}
double tgk_py_cfl_rate(void) { return g_ctx.cfl_linear; }
void tgk_py_set_dt(double dt) { g_ctx.dt = dt; g_ctx.param.dt_max = dt; }
