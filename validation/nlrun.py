"""Nonlinear s-alpha Cyclone (GX/GS2 parameters) at a sequence of resolutions."""
import sys, os, subprocess, json
TGK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tgk"))
BASE = dict(geometry="s-alpha", q0=1.4, shat=0.8, eps=0.18, rmaj=2.77778, alpha_mhd=0.0, adiabatic_electrons=1, tau_e=1.0,
            beta=0.0, nonlinear=1, imex=0, hyper_z=0.15, ky_min=0.05, z_period=1.0, vmax=3.5, dt_max=0.05, cfl_frac=0.85,
            nsteps=10**8, t_max=400.0, nprint=25, init_amp=0.01, nthreads=6,
            **{"spec0.charge": 1.0, "spec0.mass": 1.0, "spec0.dens": 1.0, "spec0.temp": 1.0,
               "spec0.fprim": 0.8 * 2.77778, "spec0.tprim": 2.49 * 2.77778, "spec0.nu": 0.002})
RES = {"R1": dict(nkx=32, nky=16, nz=24, nv=32, nmu=8), "R2": dict(nkx=64, nky=24, nz=24, nv=32, nmu=8),
       "R3": dict(nkx=128, nky=32, nz=24, nv=32, nmu=8), "R4": dict(nkx=192, nky=48, nz=24, nv=32, nmu=8),
       "R3v": dict(nkx=128, nky=32, nz=24, nv=48, nmu=12), "R3z": dict(nkx=128, nky=32, nz=36, nv=32, nmu=8)}
def go(name, **extra):
    p = dict(BASE); p.update(RES[name]); p.update(extra)
    p["out_prefix"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nl", name)
    inp = p["out_prefix"] + ".in"
    open(inp, "w").write("".join(f"{k} = {v}\n" for k, v in p.items()))
    with open(p["out_prefix"] + ".log", "w") as log:
        subprocess.run([TGK, inp], stdout=log, stderr=subprocess.STDOUT)
if __name__ == "__main__":
    for n in sys.argv[1:]:
        go(n)
