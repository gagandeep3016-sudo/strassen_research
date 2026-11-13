"""Reference algorithms for the PRISM Strassen/block-LU study.
All recursive matrices are square and power-of-two sized.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import solve_triangular


def classical_mm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B


def strassen_mm(A: np.ndarray, B: np.ndarray, cutoff: int = 64) -> np.ndarray:
    n = A.shape[0]
    if A.ndim != 2 or B.ndim != 2 or A.shape != B.shape or n != A.shape[1]:
        raise ValueError("A and B must be equally sized square matrices")
    if n & (n - 1):
        raise ValueError("matrix order must be a power of two")
    if cutoff < 1:
        raise ValueError("cutoff must be positive")
    if n <= cutoff:
        return A @ B
    h = n // 2
    A11, A12 = A[:h, :h], A[:h, h:]
    A21, A22 = A[h:, :h], A[h:, h:]
    B11, B12 = B[:h, :h], B[:h, h:]
    B21, B22 = B[h:, :h], B[h:, h:]
    M1 = strassen_mm(A11 + A22, B11 + B22, cutoff)
    M2 = strassen_mm(A21 + A22, B11, cutoff)
    M3 = strassen_mm(A11, B12 - B22, cutoff)
    M4 = strassen_mm(A22, B21 - B11, cutoff)
    M5 = strassen_mm(A11 + A12, B22, cutoff)
    M6 = strassen_mm(A21 - A11, B11 + B12, cutoff)
    M7 = strassen_mm(A12 - A22, B21 + B22, cutoff)
    C = np.empty_like(A)
    C[:h, :h] = M1 + M4 - M5 + M7
    C[:h, h:] = M3 + M5
    C[h:, :h] = M2 + M4
    C[h:, h:] = M1 - M2 + M3 + M6
    return C


def lu_unpivoted(A: np.ndarray):
    """Doolittle LU without pivoting; returns unit-lower L and upper U."""
    X = np.array(A, dtype=np.float64, copy=True)
    n = X.shape[0]
    for k in range(n - 1):
        if X[k, k] == 0.0:
            raise np.linalg.LinAlgError("zero pivot in unpivoted LU")
        X[k + 1:, k] /= X[k, k]
        X[k + 1:, k + 1:] -= np.outer(X[k + 1:, k], X[k, k + 1:])
    L = np.tril(X, -1) + np.eye(n)
    U = np.triu(X)
    return L, U


class RecursiveBlockLU:
    """Unpivoted recursive block LU with Schur-complement updates."""
    def __init__(self, block_cutoff: int = 64, strassen_cutoff: int | None = None):
        self.block_cutoff = block_cutoff
        self.strassen_cutoff = strassen_cutoff

    def mm(self, A, B):
        if self.strassen_cutoff is None:
            return A @ B
        return strassen_mm(A, B, self.strassen_cutoff)

    def trsm_left(self, L, B):  # L X = B, L unit lower
        n = L.shape[0]
        if n <= self.block_cutoff:
            return solve_triangular(L, B, lower=True, unit_diagonal=True, check_finite=False)
        h = n // 2
        L11, L21, L22 = L[:h, :h], L[h:, :h], L[h:, h:]
        Xtop = np.hstack((self.trsm_left(L11, B[:h, :h]),
                          self.trsm_left(L11, B[:h, h:])))
        R = B[h:, :] - np.hstack((self.mm(L21, Xtop[:, :h]),
                                  self.mm(L21, Xtop[:, h:])))
        Xbottom = np.hstack((self.trsm_left(L22, R[:, :h]),
                             self.trsm_left(L22, R[:, h:])))
        return np.vstack((Xtop, Xbottom))

    def trsm_right(self, U, B):  # X U = B, U upper
        n = U.shape[0]
        if n <= self.block_cutoff:
            return solve_triangular(U.T, B.T, lower=True, check_finite=False).T
        h = n // 2
        U11, U12, U22 = U[:h, :h], U[:h, h:], U[h:, h:]
        Xleft = np.vstack((self.trsm_right(U11, B[:h, :h]),
                           self.trsm_right(U11, B[h:, :h])))
        R = B[:, h:] - np.vstack((self.mm(Xleft[:h, :], U12),
                                  self.mm(Xleft[h:, :], U12)))
        Xright = np.vstack((self.trsm_right(U22, R[:h, :]),
                            self.trsm_right(U22, R[h:, :])))
        return np.hstack((Xleft, Xright))

    def factor(self, A):
        n = A.shape[0]
        if n <= self.block_cutoff:
            return lu_unpivoted(A)
        h = n // 2
        A11, A12 = A[:h, :h], A[:h, h:]
        A21, A22 = A[h:, :h], A[h:, h:]
        L11, U11 = self.factor(A11)
        U12 = self.trsm_left(L11, A12)
        L21 = self.trsm_right(U11, A21)
        S = A22 - self.mm(L21, U12)
        L22, U22 = self.factor(S)
        L = np.zeros_like(A); U = np.zeros_like(A)
        L[:h, :h], L[h:, :h], L[h:, h:] = L11, L21, L22
        U[:h, :h], U[:h, h:], U[h:, h:] = U11, U12, U22
        return L, U


def solve_from_lu(L, U, b):
    y = solve_triangular(L, b, lower=True, unit_diagonal=True, check_finite=False)
    return solve_triangular(U, y, lower=False, check_finite=False)


def relative_residual(A, x, b):
    den = np.linalg.norm(A, 2) * np.linalg.norm(x, 2) + np.linalg.norm(b, 2)
    return np.linalg.norm(A @ x - b, 2) / den
