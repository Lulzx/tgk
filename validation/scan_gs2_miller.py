import gs2run, json
ky = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
out = {}
for kappa in [1.0, 1.4, 1.8]:
    r = gs2run.run(f"miller_k{kappa}", f"/Users/lulzx/work/gs2build/runs/miller_k{kappa}", ky_tgk=ky, geometry="miller",
                   kappa=kappa, ntheta=48, nperiod=5, negrid=24, ngauss=12, delt=0.02, nstep=20000, nproc=6)
    out[str(kappa)] = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in r.items()}
    json.dump(out, open("results/gs2_miller.json", "w"), indent=1)
    print(kappa, out[str(kappa)], flush=True)
