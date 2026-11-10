"""Largest-|lambda| eigenvalues of the semi-discrete TGK operator L ~ (P(dt)-I)/dt, dt tiny."""
import ctypes, numpy as np, math, os, tempfile, sys, json
from scipy.sparse.linalg import LinearOperator, eigs
import tgkrun
def spec(ky, k=4, **over):
    lib = ctypes.CDLL(os.path.abspath("libtgk_fp64.dylib"))
    prm = dict(tgkrun.CBC); prm.update(over); prm["ky_min"] = ky
    prm.setdefault("kx_max", 2*math.pi*abs(prm["shat"])*ky*(prm["nkx"]//2))
    wd = tempfile.mkdtemp(dir=os.environ["TGK_SCRATCH"]); prm["out_prefix"] = wd+"/a"; tgkrun.write_input(wd+"/a.in", prm)
    lib.tgk_py_init.restype = ctypes.c_long; lib.tgk_py_cfl_rate.restype = ctypes.c_double
    n = lib.tgk_py_init((wd+"/a.in").encode()); rate = lib.tgk_py_cfl_rate()
    dims = (ctypes.c_int*6)(); lib.tgk_py_dims(dims); dims = list(dims)
    eps = 1e-7; lib.tgk_py_set_dt(ctypes.c_double(eps))
    bi = np.zeros(n, np.complex128); bo = np.zeros(n, np.complex128)
    def mv(x):
        bi[:] = x.ravel(); lib.tgk_py_set_dt(ctypes.c_double(eps))
        lib.tgk_py_apply(bi.ctypes.data_as(ctypes.c_void_p), bo.ctypes.data_as(ctypes.c_void_p), 1)
        return (bo - bi) / eps
    lam, vec = eigs(LinearOperator((n, n), matvec=mv, dtype=np.complex128), k=k, which="LM", ncv=40, tol=1e-6, maxiter=5000)
    out = []
    for i in np.argsort(-abs(lam)):
        a = np.abs(vec[:, i].reshape(dims))**2
        out.append((lam[i], a.sum(axis=(0,1,2,3,4)).argmax(), a.sum(axis=(0,2,3,4,5)).argmax(), a.sum(axis=(0,1,3,4,5)).argmax()))
    return rate, dims, out
if __name__ == "__main__":
    over = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    rate, dims, out = spec(float(sys.argv[1]), **over)
    print("dims", dims, " CFL rate estimate", round(rate, 2), " dt_used = 0.85*2.8/rate =", round(0.85*2.8/rate, 5))
    for lam, ikx, imu, iv in out:
        print(f"  lambda = {lam.real:+.3f} {lam.imag:+.3f}i  |lambda| = {abs(lam):.2f}  kx-col {ikx} mu-node {imu} v-idx {iv}")
