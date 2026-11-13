"""Single-threaded solve benchmark and accuracy experiment.
Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python solve_bench.py
"""
import csv, os, platform, statistics, time
import numpy as np
import scipy
from scipy.linalg import solve
from threadpoolctl import threadpool_limits, threadpool_info
from algorithms import RecursiveBlockLU, lu_unpivoted, solve_from_lu, relative_residual


def make_dd(n, seed):
    rng = np.random.default_rng(seed)
    R = rng.uniform(-1.0, 1.0, (n, n))
    return R + n * np.eye(n)


def timed(fn, repeats=5):
    fn()  # warm-up
    ts=[]
    for _ in range(repeats):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return statistics.median(ts), ts


def main():
    rows=[]
    with threadpool_limits(limits=1):
        for n in (64, 128, 256, 512, 1024):
            A=make_dd(n, 20261112+n)
            xtrue=np.ones(n); b=A@xtrue
            methods={
                "GE_unpivoted": lambda: lu_unpivoted(A),
                "blockLU_classical": lambda: RecursiveBlockLU(64,None).factor(A),
                "blockLU_Strassen_m64": lambda: RecursiveBlockLU(64,64).factor(A),
                "blockLU_Strassen_m32": lambda: RecursiveBlockLU(64,32).factor(A),
            }
            for name, factor in methods.items():
                try:
                    def full():
                        L,U=factor()
                        return solve_from_lu(L,U,b)
                    med, samples=timed(full, repeats=15)
                    L,U=factor(); x=solve_from_lu(L,U,b)
                    rows.append(dict(n=n,method=name,median_total_s=med,
                                     samples_s=";".join(f"{v:.9g}" for v in samples),
                                     relative_residual=relative_residual(A,x,b),
                                     relative_forward_error=np.linalg.norm(x-xtrue)/np.linalg.norm(xtrue),
                                     factor_reconstruction=np.linalg.norm(A-L@U)/np.linalg.norm(A)))
                except Exception as e:
                    rows.append(dict(n=n,method=name,median_total_s="",samples_s="",error=str(e)))
            med, samples=timed(lambda: solve(A,b,assume_a='gen',check_finite=False), repeats=15)
            x=solve(A,b,assume_a='gen',check_finite=False)
            rows.append(dict(n=n,method="LAPACK_pivoted_reference",median_total_s=med,
                             samples_s=";".join(f"{v:.9g}" for v in samples),
                             relative_residual=relative_residual(A,x,b),
                             relative_forward_error=np.linalg.norm(x-xtrue)/np.linalg.norm(xtrue),
                             factor_reconstruction=""))
    out=os.path.join(os.path.dirname(__file__),"..","results","solve_benchmark.csv")
    with open(out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader(); w.writerows(rows)
    print("platform:",platform.platform())
    print("processor:",platform.processor())
    print("python:",platform.python_version(),"numpy:",np.__version__,"scipy:",scipy.__version__)
    print("threadpools:",threadpool_info())
    for r in rows: print(r)
    print("saved:",os.path.abspath(out))

if __name__=="__main__": main()
