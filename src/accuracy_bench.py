"""Reproducible small accuracy study for product and unpivoted block LU."""
import csv, os, statistics
import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits
from algorithms import strassen_mm, RecursiveBlockLU, lu_unpivoted, solve_from_lu, relative_residual


def matrix(kind,n,seed):
    rng=np.random.default_rng(seed)
    if kind=="DD": return rng.uniform(-1,1,(n,n))+n*np.eye(n)
    if kind=="SPD":
        Q,_=np.linalg.qr(rng.standard_normal((n,n)))
        vals=np.logspace(0,-6,n)
        return (Q*vals)@Q.T
    if kind=="GAUSS": return rng.standard_normal((n,n))
    raise ValueError(kind)


def main():
    rows=[]
    with threadpool_limits(limits=1):
        # Product error against extended-precision reference.
        for n in (64,128,256):
            rng=np.random.default_rng(700+n)
            A=rng.uniform(-1,1,(n,n));B=rng.uniform(-1,1,(n,n))
            ref=A.astype(np.longdouble)@B.astype(np.longdouble)
            denom=float(np.linalg.norm(A)*np.linalg.norm(B))
            for cutoff in (8,16,32,64):
                if cutoff>=n: continue
                C=strassen_mm(A,B,cutoff).astype(np.longdouble)
                err=float(np.linalg.norm(C-ref)/denom)
                rows.append(dict(experiment="product",matrix_class="random_uniform",n=n,method="Strassen",cutoff=cutoff,
                                 relative_error=err,relative_residual="",forward_error="",condition_number=""))
        # Linear-system accuracy. Matrices DD and SPD are chosen so leading pivots exist;
        # Gaussian matrices are included as a cautionary no-pivot stress test.
        for kind in ("DD","SPD","GAUSS"):
            for n in (64,128,256):
                for seed in (11,29,47):
                    A=matrix(kind,n,seed+n)
                    xtrue=np.ones(n);b=A@xtrue
                    kappa=float(np.linalg.cond(A))
                    configs=[("GE_unpivoted",None),
                             ("blockLU_classical",RecursiveBlockLU(32,None)),
                             ("blockLU_Strassen_m32",RecursiveBlockLU(32,32)),
                             ("blockLU_Strassen_m8",RecursiveBlockLU(8,8))]
                    for name,cfg in configs:
                        try:
                            L,U=lu_unpivoted(A) if cfg is None else cfg.factor(A)
                            x=solve_from_lu(L,U,b)
                            rows.append(dict(experiment="solve",matrix_class=kind,n=n,method=name,cutoff="",
                                relative_error=float(np.linalg.norm(A-L@U)/np.linalg.norm(A)),
                                relative_residual=float(relative_residual(A,x,b)),
                                forward_error=float(np.linalg.norm(x-xtrue)/np.linalg.norm(xtrue)),
                                condition_number=kappa))
                        except Exception as e:
                            rows.append(dict(experiment="solve",matrix_class=kind,n=n,method=name,cutoff="",
                                relative_error="",relative_residual="",forward_error="",condition_number=kappa,error=str(e)))
                    xp=solve(A,b,assume_a="gen",check_finite=False)
                    rows.append(dict(experiment="solve",matrix_class=kind,n=n,method="LAPACK_pivoted",cutoff="",
                        relative_error="",relative_residual=float(relative_residual(A,xp,b)),
                        forward_error=float(np.linalg.norm(xp-xtrue)/np.linalg.norm(xtrue)),condition_number=kappa))
    out=os.path.join(os.path.dirname(__file__),"..","results","accuracy_benchmark.csv")
    with open(out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
    print("saved:",os.path.abspath(out),"rows:",len(rows))
    # concise summary
    for kind in ("DD","SPD","GAUSS"):
        print("\n",kind)
        for r in rows:
            if r.get("experiment")=="solve" and r.get("matrix_class")==kind and r.get("n")==256 and r.get("method") in ("GE_unpivoted","blockLU_classical","blockLU_Strassen_m32","blockLU_Strassen_m8","LAPACK_pivoted"):
                print(r)
if __name__=="__main__": main()
