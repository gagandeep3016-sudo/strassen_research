"""Exact arithmetic-operation recurrences used in the paper."""
from functools import lru_cache
import math

def classical_mm(n):
    return 2*n**3-n**2

@lru_cache(None)
def strassen_mm_count(n, cutoff=8):
    if n <= cutoff:
        return classical_mm(n)
    if n % 2:
        raise ValueError('recurrence assumes even dimensions')
    h=n//2
    return 7*strassen_mm_count(h,cutoff)+18*h*h

@lru_cache(None)
def unpivoted_lu_count(n):
    return sum((n-k-1)+2*(n-k-1)**2 for k in range(n))

@lru_cache(None)
def trsm_left_count(n, block_cutoff=8, mult_cutoff=8, use_strassen=True):
    if n<=block_cutoff:
        return n*n*(n-1)
    h=n//2
    mm=strassen_mm_count(h,mult_cutoff) if use_strassen else classical_mm(h)
    return 4*trsm_left_count(h,block_cutoff,mult_cutoff,use_strassen)+2*mm+2*h*h

@lru_cache(None)
def trsm_right_count(n, block_cutoff=8, mult_cutoff=8, use_strassen=True):
    if n<=block_cutoff:
        return n**3
    h=n//2
    mm=strassen_mm_count(h,mult_cutoff) if use_strassen else classical_mm(h)
    return 4*trsm_right_count(h,block_cutoff,mult_cutoff,use_strassen)+2*mm+2*h*h

@lru_cache(None)
def block_lu_count(n, block_cutoff=8, mult_cutoff=8, use_strassen=True):
    if n<=block_cutoff:
        return unpivoted_lu_count(n)
    h=n//2
    mm=strassen_mm_count(h,mult_cutoff) if use_strassen else classical_mm(h)
    return (2*block_lu_count(h,block_cutoff,mult_cutoff,use_strassen)
            +trsm_left_count(h,block_cutoff,mult_cutoff,use_strassen)
            +trsm_right_count(h,block_cutoff,mult_cutoff,use_strassen)
            +mm+h*h)


def main():
    print('n,classical_mm,strassen_m8,product_ratio,GE,blockLU_Strassen_m8,LU_ratio')
    for n in (64,128,256,512,1024):
        cm=classical_mm(n); sm=strassen_mm_count(n,8)
        ge=unpivoted_lu_count(n); bl=block_lu_count(n,8,8,True)
        print(f'{n},{cm},{sm},{sm/cm:.6f},{ge},{bl},{bl/ge:.6f}')
    omega=math.log2(7)
    gamma=lambda t:2**t/((2**t-4)*(2**t-2))
    print('omega=',omega,'gamma_classical=',gamma(3),'gamma_Strassen=',gamma(omega))
    print('full recursion first power-of-two crossover:',next(n for n in (2**k for k in range(1,20)) if strassen_mm_count(n,1)<classical_mm(n)))
    print('cutoff-8 first power-of-two crossover:',next(n for n in (2**k for k in range(1,20)) if strassen_mm_count(n,8)<classical_mm(n)))

if __name__=='__main__': main()
