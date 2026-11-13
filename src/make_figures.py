import os, sys, csv, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
RES=os.path.join(ROOT,'results'); FIG=os.path.join(ROOT,'figures'); os.makedirs(FIG,exist_ok=True)

def std(n): return 2*n**3-n**2
def s_count(n,cut=8):
    if n<=cut:return std(n)
    h=n//2
    return 7*s_count(h,cut)+18*h*h

def lu_base(n): return sum((n-k-1)+2*(n-k-1)**2 for k in range(n))
def trsmL(n,b0,m0,use_s):
    if n<=b0:return n*n*(n-1)
    h=n//2; mm=s_count(h,m0) if use_s else std(h)
    return 4*trsmL(h,b0,m0,use_s)+2*mm+2*h*h
def trsmR(n,b0,m0,use_s):
    if n<=b0:return n**3
    h=n//2; mm=s_count(h,m0) if use_s else std(h)
    return 4*trsmR(h,b0,m0,use_s)+2*mm+2*h*h
def blocklu(n,b0,m0,use_s):
    if n<=b0:return lu_base(n)
    h=n//2; mm=s_count(h,m0) if use_s else std(h)
    return 2*blocklu(h,b0,m0,use_s)+trsmL(h,b0,m0,use_s)+trsmR(h,b0,m0,use_s)+mm+h*h

# Figure 1: operation and multiplication timing crossovers
fig,ax=plt.subplots(1,2,figsize=(7.0,2.35))
ns=[2**k for k in range(3,13)]
ax[0].plot(ns,[s_count(n,8)/std(n) for n in ns],marker='o',label='Strassen, cutoff 8')
ax[0].plot(ns,[s_count(n,1)/std(n) for n in ns],marker='s',label='Full recursion to 1')
ax[0].axhline(1,color='black',lw=.8,ls='--');ax[0].set_xscale('log',base=2);ax[0].set_xticks([8,64,512,4096]);ax[0].set_xticklabels(['8','64','512','4096']);ax[0].set_xlabel('Matrix order n');ax[0].set_ylabel('Arithmetic ops / classical');ax[0].set_title('(a) Exact operation counts');ax[0].legend(fontsize=7)
m=pd.read_csv(os.path.join(RES,'multiplication_benchmark.csv'))
base=m[m.method=='numpy_classical_BLAS'].set_index('n').median_s
rat=[]; x=[]
for n,g in m.groupby('n'):
    ss=g[g.method=='strassen_over_numpy_BLAS']
    if len(ss):
        best=ss.loc[ss.median_s.idxmin()]
        x.append(n);rat.append(best.median_s/base.loc[n])
ax[1].plot(x,rat,marker='o',label='Best tested Strassen cutoff')
ax[1].axhline(1,color='black',lw=.8,ls='--');ax[1].set_xscale('log',base=2);ax[1].set_xticks(x);ax[1].set_xticklabels([str(v) for v in x]);ax[1].set_xlabel('Matrix order n');ax[1].set_ylabel('Median time ratio');ax[1].set_title('(b) Strassen over single-thread BLAS');ax[1].legend(fontsize=7)
fig.tight_layout();fig.savefig(os.path.join(FIG,'fig1_multiplication.pdf'));fig.savefig(os.path.join(FIG,'fig1_multiplication.png'),dpi=220);plt.close(fig)

# Figure 2: block LU operation counts and complete solve timings
fig,ax=plt.subplots(1,2,figsize=(7.0,2.35))
ns=[2**k for k in range(6,11)]
ax[0].plot(ns,[blocklu(n,n,0,False)/lu_base(n) for n in ns],marker='s',label='Block LU, classical products')
ax[0].plot(ns,[blocklu(n,8,8,True)/lu_base(n) for n in ns],marker='o',label='Block LU, Strassen cutoff 8')
ax[0].axhline(1,color='black',lw=.8,ls='--');ax[0].set_xscale('log',base=2);ax[0].set_xticks(ns);ax[0].set_xticklabels([str(v) for v in ns]);ax[0].set_xlabel('Matrix order n');ax[0].set_ylabel('Arithmetic ops / unblocked GE');ax[0].set_title('(a) Exact LU operation counts');ax[0].legend(fontsize=6.5)
s=pd.read_csv(os.path.join(RES,'solve_benchmark.csv'))
ns=sorted(n for n in s.n.unique() if n>=256); ratios=[]
for n in ns:
 g=s[s.n==n].set_index('method')
 ratios.append(float(g.loc['blockLU_Strassen_m64','median_total_s'])/float(g.loc['blockLU_classical','median_total_s']))
ax[1].plot(ns,ratios,marker='o',label='Strassen m=64 / classical block LU')
ax[1].axhline(1,color='black',lw=.8,ls='--');ax[1].set_xscale('log',base=2);ax[1].set_xticks(ns);ax[1].set_xticklabels([str(v) for v in ns]);ax[1].set_xlabel('Matrix order n');ax[1].set_ylabel('Median total-solve time ratio');ax[1].set_title('(b) Full solve, same block framework');ax[1].legend(fontsize=6.5)
fig.tight_layout();fig.savefig(os.path.join(FIG,'fig2_blocklu.pdf'));fig.savefig(os.path.join(FIG,'fig2_blocklu.png'),dpi=220);plt.close(fig)

# Figure 3: residual comparison on representative n=256 matrix classes
acc=pd.read_csv(os.path.join(RES,'accuracy_benchmark.csv'))
methods=['GE_unpivoted','blockLU_classical','blockLU_Strassen_m32','blockLU_Strassen_m8','LAPACK_pivoted']
labels=['GE no pivot','Block classical','Block Strassen 32','Block Strassen 8','LAPACK pivoted']
fig,ax=plt.subplots(1,1,figsize=(7.0,2.8))
width=.15; xpos=np.arange(3)
for i,meth in enumerate(methods):
    vals=[]
    for cls in ['DD','SPD','GAUSS']:
        g=acc[(acc.experiment=='solve')&(acc.matrix_class==cls)&(acc.n==256)&(acc.method==meth)]
        vals.append(float(g.relative_residual.median()))
    ax.bar(xpos+(i-2)*width,vals,width,label=labels[i])
ax.set_yscale('log');ax.set_xticks(xpos);ax.set_xticklabels(['DD, κ≈1.1','SPD, κ=10⁶','Gaussian, no pivot'])
ax.set_ylabel('Median scaled residual (log scale)');ax.set_title('Accuracy of solving Ax=b at n=256 (3 seeds)');ax.legend(fontsize=6.5,ncol=3)
fig.tight_layout();fig.savefig(os.path.join(FIG,'fig3_accuracy.pdf'));fig.savefig(os.path.join(FIG,'fig3_accuracy.png'),dpi=220);plt.close(fig)
print('Figures saved to',FIG)
