import ctypes, numpy as np, math, os, tempfile, sys, json
from scipy.sparse.linalg import LinearOperator, eigs
import tgkrun
def dom(ky, dt, T=2.0, k=3, **over):
    lib = ctypes.CDLL(os.path.abspath("libtgk_fp64.dylib"))
    prm = dict(tgkrun.CBC); prm.update(over); prm["ky_min"] = ky
    prm.setdefault("kx_max", 2*math.pi*abs(prm["shat"])*ky*(prm["nkx"]//2))
    wd = tempfile.mkdtemp(dir=os.environ["TGK_SCRATCH"]); prm["out_prefix"] = wd+"/a"; tgkrun.write_input(wd+"/a.in", prm)
    lib.tgk_py_init.restype = ctypes.c_long
    n = lib.tgk_py_init((wd+"/a.in").encode())
    dims = (ctypes.c_int*6)(); lib.tgk_py_dims(dims); dims = list(dims)
    ns = max(1, int(round(T/dt)))
    bi = np.zeros(n, np.complex128); bo = np.zeros(n, np.complex128)
    def mv(x):
        bi[:] = x.ravel(); lib.tgk_py_set_dt(ctypes.c_double(dt))
        lib.tgk_py_apply(bi.ctypes.data_as(ctypes.c_void_p), bo.ctypes.data_as(ctypes.c_void_p), ns)
        return bo.copy()
    mu, vec = eigs(LinearOperator((n, n), matvec=mv, dtype=np.complex128), k=k, which="LM", ncv=30, tol=1e-7, maxiter=3000)
    lam = np.log(mu)/(ns*dt)
    res = []
    for i in np.argsort(-lam.real):
        a = np.abs(vec[:, i].reshape(dims))**2
        res.append((lam[i].real, -lam[i].imag, int(a.sum(axis=(0,1,2,3,4)).argmax()), int(a.sum(axis=(0,2,3,4,5)).argmax())))
    return res
if __name__ == "__main__":
    ky = float(sys.argv[1]); over = json.loads(sys.argv[2])
    for dt in [float(x) for x in sys.argv[3].split(",")]:
        r = dom(ky, dt, **over)
        print(f"dt={dt}: " + "; ".join(f"g={g:+.4f} w={w:+.3f} kx{kx} mu{m}" for g,w,kx,m in r), flush=True)
