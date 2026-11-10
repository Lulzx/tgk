"""TGK kinetic-electron runs on GX benchmark parameters (circular Miller, CBC gradients)."""
import tgkrun
KE = {"geometry": "miller", "nspecies": 2, "adiabatic_electrons": 0, "nmu": 8, "nv": 32, "nz": 24,
      "spec1.charge": -1.0, "spec1.mass": 2.7e-4, "spec1.dens": 1.0, "spec1.temp": 1.0,
      "spec1.fprim": 0.8 * 2.77778, "spec1.tprim": 2.49 * 2.77778, "spec1.nu": 0.0}
def run(ky, beta, imex, t_max=40.0, **o):
    p = dict(KE); p.update(beta=beta, imex=imex); p.update(o)
    if not imex: p.setdefault("dt_max", 0.01)
    return tgkrun.run(ky, t_max=t_max, **p)
