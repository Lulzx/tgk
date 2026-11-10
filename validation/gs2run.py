"""Generate, run and read GS2 linear ky scans. Results are converted to TGK/GX units
(v_t = sqrt(T/m), rho = v_t/Omega, L_ref = a): GS2 uses v_ref = sqrt(2T/m), so
ky_TGK = ky_GS2 / sqrt(2) and omega_TGK = sqrt(2) * omega_GS2."""
import os, subprocess, math, numpy as np
from netCDF4 import Dataset

GS2 = "/Users/lulzx/work/gs2build/gs2/bin/gs2"
S2 = math.sqrt(2.0)

def species_block(i, z, mass, tprim, fprim, vnewk, typ):
    return f"""&species_parameters_{i}
 z = {z} mass = {mass} dens = 1.0 temp = 1.0 tprim = {tprim} fprim = {fprim} uprim = 0.0 vnewk = {vnewk} type = '{typ}'
/
&dist_fn_species_knobs_{i}
 fexpr = 0.48 bakdif = 0.05
/
"""

def make_input(ky_tgk, geometry="s-alpha", kappa=1.0, s_kappa=0.0, tri=0.0, s_tri=0.0,
               shift=0.0, kinetic_e=False, beta=0.0, mass_e=2.7e-4, tprim=2.49, fprim=0.8,
               tprim_e=None, fprim_e=None, ntheta=32, nperiod=4, negrid=16, ngauss=8,
               delt=0.02, nstep=12000, q=1.4, shat=0.8, rmaj=2.77778, eps=0.18, vnewk=0.0,
               beta_prime=0.0):
    aky = np.atleast_1d(ky_tgk) * S2
    if geometry == "s-alpha":
        geo = f"""&theta_grid_knobs
 equilibrium_option = 's-alpha'
/
&theta_grid_salpha_knobs
 model_option = 'default'
/
&theta_grid_parameters
 ntheta = {ntheta} nperiod = {nperiod} eps = {eps} epsl = {2.0/rmaj} shat = {shat} pk = {2.0/(q*rmaj)} shift = 0.0
/
"""
    else:
        rhoc = eps * rmaj
        geo = f"""&theta_grid_knobs
 equilibrium_option = 'eik'
/
&theta_grid_parameters
 ntheta = {ntheta} nperiod = {nperiod} rhoc = {rhoc} qinp = {q} shat = {shat} rmaj = {rmaj} r_geo = {rmaj}
 shift = {shift} akappa = {kappa} akappri = {s_kappa} tri = {tri} tripri = {s_tri}
/
&theta_grid_eik_knobs
 local_eq = .true. irho = 2 bishop = 4 s_hat_input = {shat} beta_prime_input = {beta_prime} equal_arc = .false.
 iflux = 0 delrho = 1e-3 isym = 0 writelots = .false.
/
"""
    nspec = 2 if kinetic_e else 1
    spec = species_block(1, 1.0, 1.0, tprim, fprim, vnewk, "ion")
    if kinetic_e:
        spec += species_block(2, -1.0, mass_e, tprim if tprim_e is None else tprim_e,
                              fprim if fprim_e is None else fprim_e, 0.0, "electron")
    fapar = 1.0 if beta > 0 else 0.0
    ky_list = " ".join(f"{a:.6f}" for a in aky)
    return f"""&kt_grids_knobs
 grid_option = 'range'
/
&kt_grids_range_parameters
 naky = {len(aky)} aky_min = {aky.min():.6f} aky_max = {aky.max():.6f} ntheta0 = 1 theta0_min = 0.0 theta0_max = 0.0
/
{geo}
&le_grids_knobs
 ngauss = {ngauss} negrid = {negrid}
/
&dist_fn_knobs
 boundary_option = 'linked' adiabatic_option = 'iphi00=2'
/
&fields_knobs
 field_option = 'implicit'
/
&knobs
 wstar_units = .false. fphi = 1.0 fapar = {fapar} fbpar = 0.0 beta = {beta} tite = 1.0 zeff = 1.0
 delt = {delt} nstep = {nstep}
/
&reinit_knobs
 delt_adj = 2.0 delt_minimum = 1e-6
/
&collisions_knobs
 collision_model = '{'none' if vnewk == 0 else 'default'}'
/
&nonlinear_terms_knobs
 nonlinear_mode = 'off'
/
&species_knobs
 nspec = {nspec}
/
{spec}
&init_g_knobs
 ginit_option = 'noise' phiinit = 1e-3 chop_side = .false. constant_random_flag = .true.
/
&gs2_diagnostics_knobs
 nwrite = 50 navg = 50 omegatol = -1e-3 omegatinst = 500.0 save_for_restart = .false.
 write_omega = .true. write_fluxes = .false. write_ascii = .false. write_final_fields = .false.
/
""", aky

def run(name, workdir, nproc=10, **kw):
    os.makedirs(workdir, exist_ok=True)
    txt, aky = make_input(**kw)
    inp = os.path.join(workdir, name + ".in")
    with open(inp, "w") as f:
        f.write(txt)
    r = subprocess.run(["mpirun", "-np", str(nproc), GS2, inp], capture_output=True, text=True, cwd=workdir)
    if r.returncode != 0:
        raise RuntimeError(r.stdout[-3000:] + r.stderr[-3000:])
    return read(os.path.join(workdir, name + ".out.nc"))

def read(path, frac=0.2):
    d = Dataset(path)
    t = d["t"][:]
    ky = d["ky"][:]
    om = d["omega"][:]          # (t, ky, kx, ri)
    n = max(3, int(frac * len(t)))
    w = om[-n:, :, 0, 0].mean(axis=0); g = om[-n:, :, 0, 1].mean(axis=0)
    gs = om[-n:, :, 0, 1].std(axis=0)
    return dict(ky=np.array(ky) / S2, omega=np.array(w) * S2, gamma=np.array(g) * S2,
                gamma_spread=np.array(gs) * S2, t_end=float(t[-1]) / S2)
