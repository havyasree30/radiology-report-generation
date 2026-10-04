| Group | Metric | Validation | Test | Test minus validation |
|---|---:|---:|---:|---:|
| classification (IU, same pipeline) | macro AUROC | 0.847 | 0.830 | -0.017 |
| classification (IU, same pipeline) | macro AUPRC | 0.465 | 0.374 | -0.091 |
| classification (IU, same pipeline) | macro F1 | 0.330 | 0.312 | -0.018 |
| classification (IU, same pipeline) | Brier score (macro) | 0.054 | 0.054 | 0.000 |
| classification (IU, same pipeline) | ECE (macro) | 0.054 | 0.051 | -0.003 |
| retrieval (classifier query) | Jaccard@5 | 0.557 | 0.558 | 0.002 |
| retrieval (classifier query) | finding coverage@5 | 0.748 | 0.738 | -0.010 |
| report generation | finding precision | 0.411 | 0.413 | 0.002 |
| report generation | finding recall | 0.358 | 0.306 | -0.052 |
| report generation | finding F1 | 0.383 | 0.351 | -0.031 |
| report generation | hallucination rate | 0.257 | 0.217 | -0.039 |
| report generation | omission rate | 0.381 | 0.364 | -0.017 |
| report generation | classifier FP propagation | 0.405 | 0.356 | -0.049 |
| report generation | classifier TP retention | 0.718 | 0.702 | -0.016 |
| system | decision coverage | 0.937 | 0.891 | -0.045 |
| system | normal recall (decided) | 0.781 | 0.740 | -0.041 |
| system | abnormal recall (decided) | 0.760 | 0.735 | -0.026 |
| system | indeterminate rate (P3) | 0.090 | 0.115 | 0.025 |
| system | abnormal-routing -> normal-prose rate (%) | 31.004 | 36.607 | 5.603 |
