#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
cd src
python validate.py
python theory_counts.py
python multiplication_bench.py
python solve_bench.py
python accuracy_bench.py
python make_figures.py
cd ..
pdflatex -interaction=nonstopmode -halt-on-error paper.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error paper.tex >/dev/null
pdfinfo paper.pdf | grep -E 'Pages|Page size'
