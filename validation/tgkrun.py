"""Run TGK linear single-ky cases and extract gamma, omega from the trace."""
import subprocess, os, sys, math, tempfile
import numpy as np

TGK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tgk")

CBC = dict(  # GX/GS2 Cyclone parameters (Lref = a)
    geometry="s-alpha", q0=1.4, shat=0.8, eps=0.18, rmaj=2.77778, alpha_mhd=0.0,
    adiabatic_electrons=1, tau_e=1.0, beta=0.0, nonlinear=0, imex=0, hyper_z=0.15,
    nkx=8, nky=2, nz=24, nv=32, nmu=8, vmax=3.5, z_period=1.0,
    dt_max=0.05, cfl_frac=0.85, nprint=20, init_amp=1e-6, nsteps=10**7,
    **{"spec0.charge": 1.0, "spec0.mass": 1.0, "spec0.dens": 1.0, "spec0.temp": 1.0,
       "spec0.fprim": 0.8 * 2.77778, "spec0.tprim": 2.49 * 2.77778, "spec0.nu": 0.0})

def write_input(path, prm):
    with open(path, "w") as f:
        for k, v in prm.items():
            f.write(f"{k} = {v}\n")

def run(ky, t_max=None, workdir=None, **over):
    prm = dict(CBC); prm.update(over)
    prm["ky_min"] = ky
    if "kx_max" not in over:
        # dkx = 2 pi shat ky: every kx is linked to its neighbour (one ballooning chain)
        prm["kx_max"] = 2 * math.pi * abs(prm["shat"]) * ky * prm["z_period"] * (prm["nkx"] // 2)
    if t_max is None:
        t_max = 150.0
    prm["t_max"] = t_max
    wd = workdir or tempfile.mkdtemp(prefix="tgk_", dir=os.environ.get("TGK_SCRATCH"))
    tag = f"ky{ky:.3f}"
    prm["out_prefix"] = os.path.join(wd, tag)
    inp = os.path.join(wd, tag + ".in")
    write_input(inp, prm)
    r = subprocess.run([TGK, inp], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    d = np.loadtxt(prm["out_prefix"] + ".trace.txt", comments="#")
    t, phi2, gam, om = d[:, 1], d[:, 3], d[:, 8], d[:, 9]
    # Fit over the last 40% of the run
    sel = t > 0.6 * t[-1]
    gfit = np.polyfit(t[sel], 0.5 * np.log(phi2[sel]), 1)[0]
    return dict(ky=ky, gamma=gfit, gamma_inst=gam[sel].mean(), gamma_spread=gam[sel].std(),
                omega=om[sel].mean(), omega_spread=om[sel].std(), t_end=t[-1])

if __name__ == "__main__":
    import json, time
    t0 = time.time()
    print(run(float(sys.argv[1]) if len(sys.argv) > 1 else 0.3), "%.1fs" % (time.time() - t0))
