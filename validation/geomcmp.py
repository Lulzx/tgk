"""Compare TGK Miller geometry coefficients to GS2's eik (bishop=4, equal_arc) output."""
import sys, os, json, subprocess, numpy as np
from netCDF4 import Dataset
import gs2run
S = "/private/tmp/claude-501/-Users-lulzx-work-tgk/6a0221b5-7095-4978-8293-d0dafa15348e/scratchpad"

def gs2_geom(name, **kw):
    wd = f"/Users/lulzx/work/gs2build/runs/geom_{name}"; os.makedirs(wd, exist_ok=True)
    txt, _ = gs2run.make_input([0.3], geometry="miller", nstep=2, ntheta=64, nperiod=2, **kw)
    txt = txt.replace("equal_arc = .false.", "equal_arc = .true.")
    open(f"{wd}/g.in", "w").write(txt)
    r = subprocess.run(["mpirun", "-np", "1", gs2run.GS2, "g.in"], cwd=wd, capture_output=True, text=True)
    d = Dataset(f"{wd}/g.out.nc")
    keys = ["bmag", "gds2", "gds21", "gds22", "cvdrift", "gbdrift", "cvdrift0", "grho"]
    out = {k: np.array(d[k][:]) for k in keys}
    out["theta"] = np.array(d["theta"][:]); out["gradpar"] = np.array(d["gradpar"][:])
    return out

def tgk_geom(**kw):
    m = dict(geometry="miller", nz=96, z_period=3, q0=1.4, shat=0.8, eps=0.18, rmaj=2.77778)
    m.update(kw)
    open(f"{S}/g.in", "w").write("".join(f"{k} = {v}\n" for k, v in m.items()))
    t = np.loadtxt(subprocess.run(["./geomdump", f"{S}/g.in"], capture_output=True, text=True).stdout.splitlines(), comments="#")
    sh = m["shat"]
    return dict(z=t[:, 0], gradpar=t[0, 3], bmag=t[:, 1], gds2=t[:, 6], gds21=-sh * t[:, 5], gds22=sh * sh * t[:, 4],
                cvdrift=-2 * t[:, 8], gbdrift=-2 * t[:, 10], cvdrift0=2 * sh * t[:, 7])

def compare(name, tgk_kw, gs2_kw):
    g = gs2_geom(name, **gs2_kw); t = tgk_geom(**tgk_kw)
    print(f"== {name}: gradpar GS2 {g['gradpar'].mean():.5f}  TGK {t['gradpar']:.5f}")
    sel = np.abs(t["z"]) <= g["theta"].max()
    res = {}
    for k in ["bmag", "gds2", "gds21", "gds22", "cvdrift", "gbdrift", "cvdrift0"]:
        ref = np.interp(t["z"][sel], g["theta"], g[k])
        rel = np.max(np.abs(t[k][sel] - ref)) / np.max(np.abs(ref))
        res[k] = rel
        print(f"   {k:9s} max|TGK-GS2|/max|GS2| = {rel:.2e}")
    return res

if __name__ == "__main__":
    allres = {}
    allres["circular"] = compare("circular", {}, {})
    allres["shaped"] = compare("shaped", dict(kappa=1.6, s_kappa=0.2, delta=0.3, s_delta=0.25, shift=-0.15),
                               dict(kappa=1.6, s_kappa=0.2 * 1.6 / 0.5, tri=0.3, s_tri=0.25 / 0.5, shift=-0.15))
    json.dump(allres, open("results/geom_vs_gs2.json", "w"), indent=1)
