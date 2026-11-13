# PRISM Strassen / Block LU research package

## What this project contains

This is a reproducible research package for the paper:

**Beyond n³: Strassen’s Multiplication, Block LU Decomposition, and the Cost of Solving Ax = b**

The paper is `paper.pdf`; the editable LaTeX source is `paper.tex`.

### Project structure

```text
strassen_research/
├── paper.pdf
├── paper.tex
├── requirements.txt
├── README.md
├── src/
│   ├── algorithms.py
│   ├── theory_counts.py
│   ├── validate.py
│   ├── multiplication_bench.py
│   ├── solve_bench.py
│   ├── accuracy_bench.py
│   └── make_figures.py
├── results/
│   ├── multiplication_benchmark.csv
│   ├── solve_benchmark.csv
│   ├── accuracy_benchmark.csv
│   └── solve_benchmark.log
├── figures/
│   ├── fig1_multiplication.pdf/png
│   ├── fig2_blocklu.pdf/png
│   └── fig3_accuracy.pdf/png
└── docs/
    ├── AUDIT_NOTES.md
    └── rendered/  (page previews)
```

## Reproduce the experiments

Use Python 3.10+ and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the correctness tests and arithmetic-count derivations:

```bash
cd src
python validate.py
python theory_counts.py
```

Run all experiments with BLAS restricted to one thread:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python multiplication_bench.py
python solve_bench.py
python accuracy_bench.py
python make_figures.py
```

Compile the paper (requires a LaTeX installation):

```bash
cd ..
pdflatex paper.tex
pdflatex paper.tex
```

## Experimental interpretation

- `multiplication_benchmark.csv` compares recursive Strassen with NumPy/OpenBLAS classical multiplication. Both use the same BLAS product at the recursion base case. This measures the overhead of the recursive Strassen wrapper against an optimized classical kernel; it is not a comparison against a hand-optimized native Strassen implementation.
- `solve_benchmark.csv` times factorization plus forward/back substitution. The two block-LU variants use the same recursive framework and triangular-solve routine. LAPACK is included as a pivoted reference.
- `accuracy_benchmark.csv` contains product errors and linear-system residual/forward-error measurements for diagonally dominant, ill-conditioned SPD and unshifted Gaussian matrix families.
- The exact operation counts in the paper come from recurrences, not measured wall-clock time.

## Important before submission

1. Replace `Student Name` and `Roll No. XXXXXXX` in `paper.tex`.
2. The current measurements were executed in a Linux container on an AMD EPYC 9V74 host, with single-thread OpenBLAS 0.3.30, Python 3.13.5, NumPy 2.3.5 and SciPy 1.17.0. The host is virtualized/shared. Report this environment accurately, or rerun the scripts on your own machine and update the paper's environment/results.
3. The block LU implementation is unpivoted. Its conclusions are limited to the stated matrix families and assumptions. Do not describe it as a production-grade replacement for pivoted LU.
4. The paper has four pages of research content and a fifth page of references. Keep that limit if editing the text.
5. The timing results are implementation- and environment-specific. The main defensible conclusion is that fewer arithmetic operations did not translate into faster execution in these particular NumPy/OpenBLAS experiments.

## Scope and contribution

The work is an experimental reinterpretation of Strassen (1969), not a claim to have invented a new multiplication or LU algorithm. It derives a leading-order block-LU cost, verifies exact finite-size arithmetic recurrences, and measures the practical difference between operation count, runtime and numerical error.
