"""Leading eigenvalues of TGK's discrete propagator via Arnoldi (FP64 build).
lambda = log(mu) / (nsteps * dt): gamma = Re, omega_TGK = -Im (phi ~ exp(-i omega t))."""
import ctypes, numpy as np, sys, os, math, json, tempfile
from scipy.sparse.linalg import LinearOperator, eigs
import tgkrun

def analyze(ky, nsteps=20, k=6, prec="fp64", **over):
    lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"libtgk_{prec}.dylib"))
    prm = dict(tgkrun.CBC); prm.update(over); prm["ky_min"] = ky
    if "kx_max" not in over:
        prm["kx_max"] = 2*math.pi*abs(prm["shat"])*ky*prm["z_period"]*(prm["nkx"]//2)
    wd = tempfile.mkdtemp(dir=os.environ.get("TGK_SCRATCH")); prm["out_prefix"] = wd + "/a"
    inp = wd + "/a.in"; tgkrun.write_input(inp, prm)
    lib.tgk_py_init.restype = ctypes.c_long; lib.tgk_py_dt.restype = ctypes.c_double
    n = lib.tgk_py_init(inp.encode()); dt = lib.tgk_py_dt()
    dims = (ctypes.c_int*6)(); lib.tgk_py_dims(dims); dims = list(dims)
    cdt = np.complex128 if prec == "fp64" else np.complex64
    buf_in = np.zeros(n, cdt); buf_out = np.zeros(n, cdt)
    def mv(x):
        buf_in[:] = x.ravel()
        lib.tgk_py_apply(buf_in.ctypes.data_as(ctypes.c_void_p), buf_out.ctypes.data_as(ctypes.c_void_p), nsteps)
        return buf_out.astype(np.complex128).copy()
    A = LinearOperator((n, n), matvec=mv, dtype=np.complex128)
    rng = np.random.default_rng(0); v0 = rng.standard_normal(n) + 1j*rng.standard_normal(n)
    mu, vec = eigs(A, k=k, which="LM", v0=v0, ncv=max(2*k+1, 30), tol=1e-8, maxiter=3000)
    lam = np.log(mu.astype(complex)) / (nsteps*dt)
    order = np.argsort(-lam.real)
    out = []
    for i in order:
        v = vec[:, i].reshape(dims)  # s, mu, v, z, ky, kx
        a = np.abs(v)**2
        out.append(dict(gamma=lam[i].real, omega=-lam[i].imag,
                        frac_kx=(a.sum(axis=(0,1,2,3,4))/a.sum()).round(3).tolist(),
                        frac_v=(a.sum(axis=(0,1,3,4,5))/a.sum()).round(3).tolist(),
                        frac_mu=(a.sum(axis=(0,2,3,4,5))/a.sum()).round(3).tolist(),
                        frac_z=(a.sum(axis=(0,1,2,4,5))/a.sum()).round(3).tolist()))
    return dt, out

if __name__ == "__main__":
    ky = float(sys.argv[1]); over = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    dt, out = analyze(ky, **over)
    print("dt =", dt)
    for o in out:
        print(f"gamma={o['gamma']:+.5f} omega={o['omega']:+.5f}")
        print("   kx:", o["frac_kx"]); print("   v :", o["frac_v"]); print("   mu:", o["frac_mu"])
