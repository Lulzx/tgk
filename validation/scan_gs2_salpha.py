import gs2run, json, numpy as np
ky = [round(0.05*i,3) for i in range(1,12)]
out = {}
for res,kw in [("std", dict(ntheta=32, nperiod=4, negrid=16, ngauss=8, delt=0.025, nstep=16000)),
               ("hi",  dict(ntheta=48, nperiod=5, negrid=24, ngauss=12, delt=0.02, nstep=20000))]:
    r = gs2run.run("salpha_"+res, "/Users/lulzx/work/gs2build/runs/salpha_"+res, ky_tgk=ky, **kw)
    out[res] = {k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in r.items()}
    json.dump(out, open("results/gs2_salpha.json","w"), indent=1)
    print(res, r, flush=True)
