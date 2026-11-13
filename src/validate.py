"""Fast correctness checks for the algorithms and mathematical example."""
import numpy as np
from algorithms import strassen_mm, RecursiveBlockLU, solve_from_lu
from theory_counts import classical_mm, strassen_mm_count, unpivoted_lu_count, block_lu_count

# Strassen worked example from the paper.
A = np.array([[1., 2.], [3., 4.]])
B = np.array([[5., 6.], [7., 8.]])
expected = np.array([[19., 22.], [43., 50.]])
assert np.array_equal(strassen_mm(A, B, 1), expected)

rng = np.random.default_rng(20261112)
for n in (4, 8, 16, 32, 64):
    X = rng.standard_normal((n, n))
    Y = rng.standard_normal((n, n))
    for cutoff in (2, 4, 8, 16):
        if cutoff < n:
            C = strassen_mm(X, Y, cutoff)
            assert np.allclose(C, X @ Y, rtol=2e-12, atol=2e-12)
            
    D = rng.uniform(-1, 1, (n, n)) + n * np.eye(n)
    for b0 in (2, 4, 8, 16):
        if b0 < n:
            # Test classical block LU
            L, U = RecursiveBlockLU(b0, None).factor(D)
            assert np.linalg.norm(D - L @ U) / np.linalg.norm(D) < 1e-12
            
            # Test Strassen block LU for power-of-two sizes with valid cutoff
            if (n & (n - 1) == 0) and (b0 >= 8) and (8 <= n // 2):
                L_s, U_s = RecursiveBlockLU(b0, 8).factor(D)
                assert np.linalg.norm(D - L_s @ U_s) / np.linalg.norm(D) < 1e-11
                
            x = solve_from_lu(L, U, D @ np.ones(n))
            assert np.linalg.norm(x - np.ones(n)) / np.sqrt(n) < 1e-10

# Verify operation count recurrences
for n in (1, 2, 4, 8, 16, 32, 64):
    assert classical_mm(n) == 2 * n**3 - n**2
    if n & (n - 1) == 0:
        assert strassen_mm_count(n, 1) >= 0
        assert unpivoted_lu_count(n) >= 0
        assert block_lu_count(n, 8, 8, True) >= 0

print('PASS: worked example, Strassen products, block LU (classical & Strassen), solves, and count recurrences verified.')