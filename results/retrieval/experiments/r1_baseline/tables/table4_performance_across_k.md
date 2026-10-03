| Retriever | Query policy | K | Jaccard@K | nDCG@K | Hit@K | Union coverage@K | Zero-overlap rate@K | Duplicate-text rate@K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense (MiniLM) | Oracle finding query | 1 | 0.797 | 0.799 | 0.698 | 0.855 | 0.075 | 0.000 |
| Dense (MiniLM) | Oracle finding query | 3 | 0.805 | 0.809 | 0.906 | 0.964 | 0.077 | 0.348 |
| Dense (MiniLM) | Oracle finding query | 5 | 0.809 | 0.813 | 0.912 | 0.971 | 0.073 | 0.418 |
| Dense (MiniLM) | Oracle finding query | 10 | 0.771 | 0.790 | 0.915 | 0.977 | 0.105 | 0.473 |
| Dense (MiniLM) | Classifier: all positives | 1 | 0.522 | 0.523 | 0.450 | 0.574 | 0.372 | 0.000 |
| Dense (MiniLM) | Classifier: all positives | 3 | 0.514 | 0.518 | 0.516 | 0.646 | 0.379 | 0.328 |
| Dense (MiniLM) | Classifier: all positives | 5 | 0.514 | 0.518 | 0.547 | 0.676 | 0.388 | 0.393 |
| Dense (MiniLM) | Classifier: all positives | 10 | 0.526 | 0.529 | 0.664 | 0.814 | 0.385 | 0.449 |
| Dense (MiniLM) | Classifier: precision-aware gated | 1 | 0.520 | 0.522 | 0.462 | 0.556 | 0.399 | 0.000 |
| Dense (MiniLM) | Classifier: precision-aware gated | 3 | 0.520 | 0.523 | 0.538 | 0.652 | 0.392 | 0.332 |
| Dense (MiniLM) | Classifier: precision-aware gated | 5 | 0.518 | 0.522 | 0.575 | 0.683 | 0.400 | 0.399 |
| Dense (MiniLM) | Classifier: precision-aware gated | 10 | 0.529 | 0.532 | 0.672 | 0.808 | 0.391 | 0.454 |
| Lexical (BM25) | Oracle finding query | 1 | 0.798 | 0.801 | 0.749 | 0.828 | 0.131 | 0.000 |
| Lexical (BM25) | Oracle finding query | 3 | 0.787 | 0.793 | 0.818 | 0.916 | 0.100 | 0.348 |
| Lexical (BM25) | Oracle finding query | 5 | 0.816 | 0.814 | 0.897 | 0.961 | 0.082 | 0.209 |
| Lexical (BM25) | Oracle finding query | 10 | 0.786 | 0.799 | 0.940 | 0.977 | 0.089 | 0.367 |
| Lexical (BM25) | Classifier: all positives | 1 | 0.539 | 0.541 | 0.493 | 0.568 | 0.385 | 0.000 |
| Lexical (BM25) | Classifier: all positives | 3 | 0.533 | 0.537 | 0.541 | 0.635 | 0.383 | 0.335 |
| Lexical (BM25) | Classifier: all positives | 5 | 0.535 | 0.539 | 0.570 | 0.663 | 0.382 | 0.203 |
| Lexical (BM25) | Classifier: all positives | 10 | 0.543 | 0.545 | 0.667 | 0.791 | 0.369 | 0.354 |
| Lexical (BM25) | Classifier: precision-aware gated | 1 | 0.515 | 0.516 | 0.493 | 0.534 | 0.441 | 0.000 |
| Lexical (BM25) | Classifier: precision-aware gated | 3 | 0.529 | 0.529 | 0.550 | 0.640 | 0.401 | 0.345 |
| Lexical (BM25) | Classifier: precision-aware gated | 5 | 0.537 | 0.536 | 0.564 | 0.674 | 0.391 | 0.211 |
| Lexical (BM25) | Classifier: precision-aware gated | 10 | 0.541 | 0.541 | 0.667 | 0.800 | 0.380 | 0.365 |
