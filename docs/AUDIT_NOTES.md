# Audit notes on the initial Claude-generated package

The initial six-page draft was not used as the final experimental authority. Its mathematical structure was useful, but several items required correction or qualification.

1. The initial C classical multiplication kernel accumulated every product into a zero-initialized output, while its counter used the conventional mathematical count `n^3` multiplications plus `n^2(n-1)` additions. Those conventions did not exactly match the executed loop. The new paper uses independently stated arithmetic recurrences and NumPy-based experiments instead.
2. The original benchmark timings used minimum timings and had limited repeats at the largest sizes. The new Python timing scripts use warm-up and medians.
3. The original BLAS/Strassen JSON had no generating script. The new `multiplication_bench.py` fully specifies and regenerates its comparison.
4. The initial C benchmark measured factorization but not the complete factorization-plus-vector-solve timing pipeline. The new `solve_bench.py` measures the complete pipeline.
5. The original draft had six pages, whereas PRISM allows four pages of paper and a fifth references page. The rebuilt PDF is five pages total.
6. The block LU method remains unpivoted. Accuracy comparisons explicitly separate the Gaussian no-pivot stress test from conclusions about Strassen alone.

The original uploaded code, CSVs and PDF remain available in the conversation attachments for comparison.
