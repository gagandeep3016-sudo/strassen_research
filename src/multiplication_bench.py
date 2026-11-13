"""Median single-thread benchmark: NumPy/BLAS classical vs recursive Strassen.
This is an experiment on Strassen wrapped around the same NumPy matmul base case,
not a comparison against a separately optimized Strassen/BLAS implementation.
"""
import csv, os, platform, statistics, time
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from algorithms import strassen_mm


def median_time(fn, repeats=7):
    fn()  # warm-up
    vals=[]
    for _ in range(repeats):
        t=time.perf_counter(); fn(); vals.append(time.perf_counter()-t)
    return statistics.median(vals), vals


def main():
    rows=[]
    with threadpool_limits(limits=1):
        for n in (32,64,128,256,512,1024):
            rng=np.random.default_rng(4100+n)
            A=rng.standard_normal((n,n)); B=rng.standard_normal((n,n))
            ref=A@B
            t0,s0=median_time(lambda:A@B, repeats=9 if n<=256 else 5)
            rows.append(dict(n=n,method="numpy_classical_BLAS",cutoff="",median_s=t0,
                             samples_s=";".join(f"{x:.9g}" for x in s0),
                             relative_error=0.0))
            for cutoff in (32,64,128):
                if cutoff>=n: continue
                f=lambda:strassen_mm(A,B,cutoff)
                t,s=median_time(f,repeats=7 if n<=256 else 5)
                C=f()
                err=np.linalg.norm(C-ref)/np.linalg.norm(ref)
                rows.append(dict(n=n,method="strassen_over_numpy_BLAS",cutoff=cutoff,median_s=t,
                                 samples_s=";".join(f"{x:.9g}" for x in s),relative_error=err))
    out=os.path.join(os.path.dirname(__file__),"..","results","multiplication_benchmark.csv")
    with open(out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print("platform:",platform.platform())
    print("numpy:",np.__version__,"threadpools:",threadpool_info())
    for r in rows: print(r)
    print("saved:",os.path.abspath(out))
if __name__=="__main__": main()
