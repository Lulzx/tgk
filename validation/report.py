"""Build comparison tables and plots from results/*.json and the GX reference files."""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from netCDF4 import Dataset
GX = "/Users/lulzx/work/gs2build/gx/benchmarks/linear/ITG_cyclone"
def gx(name):
    d = Dataset(f"{GX}/itg_{name}_correct.out.nc"); om = d["Diagnostics"]["omega_kxkyt"][:, :, 0, :]
    h = len(d["Grids"]["time"][:]) // 2
    return np.array(d["Grids"]["ky"][1:]), np.array(om[h:, 1:, 1].mean(0)), np.array(om[h:, 1:, 0].mean(0))
def load(f): return json.load(open(f"results/{f}")) if os.path.exists(f"results/{f}") else None

lines = []
# s-alpha
t = load("tgk_salpha.json"); g = load("gs2_salpha.json"); kx, gg, gw = gx("salpha_adiabatic_electrons")
lines.append("## s-alpha Cyclone, adiabatic electrons\n\n| ky rho | TGK gamma (nmu=8) | TGK gamma (refined) | GS2 gamma | GX gamma | TGK omega (refined) | GS2 omega | GX omega |\n|---|---|---|---|---|---|---|---|")
fig, ax = plt.subplots(2, 2, figsize=(10, 7))
tr = {round(r["ky"], 3): r for r in t["refined"]}; td = {round(r["ky"], 3): r for r in t["default"]}
for i, k in enumerate(kx):
    k3 = round(float(k), 3); j = int(np.argmin(abs(np.array(g["hi"]["ky"]) - k)))
    lines.append(f"| {k:.2f} | {td[k3]['gamma']:.4f} | {tr[k3]['gamma']:.4f} | {g['hi']['gamma'][j]:.4f} | {gg[i]:.4f} | {-tr[k3]['omega']:.4f} | {g['hi']['omega'][j]:.4f} | {gw[i]:.4f} |")
ks = sorted(tr)
for a, key, sgn, lab in [(ax[0, 0], "gamma", 1, r"$\gamma\,a/v_t$"), (ax[1, 0], "omega", -1, r"$\omega\,a/v_t$")]:
    a.plot(kx, gg if key == "gamma" else gw, "ks", mfc="none", label="GX")
    a.plot(g["hi"]["ky"], g["hi"][key], "o", mfc="none", label="GS2")
    a.plot(ks, [sgn * tr[k][key] for k in ks], "-", label="TGK refined")
    a.plot(ks, [sgn * td[k][key] for k in ks], ":", label="TGK nmu=8")
    a.set_ylabel(lab); a.set_title("s-alpha"); a.legend(fontsize=8)
ax[1, 0].set_xlabel(r"$k_y\rho$")
# Miller
tm = load("tgk_miller.json"); gm = load("gs2_miller.json"); mx, mg, mw = gx("miller_adiabatic_electrons")
lines.append("\n## Miller, adiabatic electrons: elongation scan (q, shat, gradients fixed)\n\n| kappa | ky rho | TGK gamma | GS2 gamma | GX gamma | TGK omega | GS2 omega | GX omega |\n|---|---|---|---|---|---|---|---|")
for kap in ["1.0", "1.4", "1.8"]:
    if not tm or kap not in tm: continue
    rows = sorted(tm[kap], key=lambda r: r["ky"])
    for r in rows:
        gs = gm.get(kap) if gm else None
        j = int(np.argmin(abs(np.array(gs["ky"]) - r["ky"]))) if gs else None
        i = int(np.argmin(abs(mx - r["ky"])))
        gxg = f"{mg[i]:.4f}" if kap == "1.0" else "-"; gxw = f"{mw[i]:.4f}" if kap == "1.0" else "-"
        lines.append(f"| {kap} | {r['ky']:.1f} | {r['gamma']:.4f} | {gs['gamma'][j]:.4f} | {gxg} | {-r['omega']:.4f} | {gs['omega'][j]:.4f} | {gxw} |" if gs else
                     f"| {kap} | {r['ky']:.1f} | {r['gamma']:.4f} | - | {gxg} | {-r['omega']:.4f} | - | {gxw} |")
    c = ax[0, 1].plot([r["ky"] for r in rows], [r["gamma"] for r in rows], "-", label=f"TGK kappa={kap}")[0].get_color()
    ax[1, 1].plot([r["ky"] for r in rows], [-r["omega"] for r in rows], "-", color=c)
    if gm and kap in gm:
        ax[0, 1].plot(gm[kap]["ky"], gm[kap]["gamma"], "o", mfc="none", color=c, label=f"GS2 kappa={kap}")
        ax[1, 1].plot(gm[kap]["ky"], gm[kap]["omega"], "o", mfc="none", color=c)
ax[0, 1].plot(mx, mg, "ks", mfc="none", label="GX kappa=1"); ax[1, 1].plot(mx, mw, "ks", mfc="none")
ax[0, 1].set_title("Miller"); ax[0, 1].legend(fontsize=7); ax[1, 1].set_xlabel(r"$k_y\rho$")
plt.tight_layout(); plt.savefig("linear_comparison.png", dpi=120)
open("results/tables.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))

# Kinetic electrons, electrostatic (circular Miller)
ke = load("tgk_ke.json")
if ke:
    d = Dataset(f"{GX}/itg_miller_kinetic_electrons_correct.out.nc"); om = d["Diagnostics"]["omega_kxkyt"][:, :, 0, :]
    h = len(d["Grids"]["time"][:]) // 2; kk = np.array(d["Grids"]["ky"][:])
    kg, kw = np.array(om[h:, :, 1].mean(0)), np.array(om[h:, :, 0].mean(0))
    kl = ["\n## Kinetic electrons, electrostatic, circular Miller\n\n| ky rho | TGK gamma | GX gamma | TGK omega | GX omega |\n|---|---|---|---|---|"]
    for r in ke:
        i = int(np.argmin(abs(kk - r["ky"])))
        kl.append(f"| {r['ky']:.1f} | {r['gamma']:.4f} | {kg[i]:.4f} | {-r['omega']:.4f} | {kw[i]:.4f} |")
    lines += kl; print("\n".join(kl))

# Nonlinear chi_i convergence
runs = [(n, f"nl/{n}.trace.txt") for n in ["R1", "R2", "R3", "R3v"] if os.path.exists(f"nl/{n}.trace.txt")]
if runs:
    grids = {"R1": "32 x 16", "R2": "64 x 24", "R3": "128 x 32", "R3v": "128 x 32 (nv 48, nmu 12)"}
    nl = ["\n## Nonlinear CBC chi_i (rho^2 v_t / a)\n\n| run | nkx x nky | t_end | <chi_i> t in [150, t_end] | SE (25-unit blocks) | <chi_i> [150, 275] | <chi_i> [275, t_end] |\n|---|---|---|---|---|---|---|"]
    def se(t, c, a, b, w=25.0):
        bl = [c[(t >= x) & (t < x + w)].mean() for x in np.arange(a, b - w + 1e-9, w)]
        return np.std(bl, ddof=1) / np.sqrt(len(bl)) if len(bl) > 1 else float("nan")
    plt.figure(figsize=(8, 4))
    for n, f in runs:
        d = np.loadtxt(f); t, c = d[:, 1], d[:, 7]
        plt.plot(t, c, lw=0.8, label=f"{n} ({grids[n]})")
        m = t >= 150; m1 = m & (t < 275); m2 = t >= 275
        av = lambda mm: f"{c[mm].mean():.3f}" if mm.sum() > 10 else "-"
        nl.append(f"| {n} | {grids[n]} | {t[-1]:.0f} | {av(m)} | {se(t, c, 150, t[-1]):.3f} | {av(m1)} | {av(m2)} |" if m.sum() > 10 else f"| {n} | {grids[n]} | {t[-1]:.0f} | - | - | - | - |")
    plt.axvspan(150, 400, color="0.9", zorder=0); plt.xlabel(r"$t\,v_t/a$"); plt.ylabel(r"$\chi_i\ [\rho^2 v_t/a]$")
    plt.ylim(0, 4); plt.legend(); plt.tight_layout(); plt.savefig("chi_convergence.png", dpi=120)
    lines += nl; print("\n".join(nl))
open("results/tables.md", "w").write("\n".join(lines) + "\n")
