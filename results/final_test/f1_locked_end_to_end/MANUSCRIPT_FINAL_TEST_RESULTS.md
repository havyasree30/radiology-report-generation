# Results: one-shot locked end-to-end evaluation

## Population

Of 578 locked studies, 550 had a frontal image, 549 formed the primary end-to-end set and 368 the clinical-finding subset (202 reference-normal) (Table 1). The guard routed 262 studies normal, 224 abnormal and 63 indeterminate (11.5%).

## Classification

The frozen classifier reached a macro AUROC of 0.830 (95% CI 0.771 to 0.876) and macro AUPRC of 0.374 (95% CI 0.353 to 0.474) over 13 defined classes (micro AUROC 0.881, micro AUPRC 0.561); macro precision 0.339, recall 0.442, specificity 0.944, F1 0.312, balanced accuracy 0.693; Brier score 0.054, log loss 0.190 and ECE 0.051 (Tables 2 and 3, Figures 1 and 2). Per-class AUROC ranged from 0.665 to 0.986; Enlarged Cardiomediastinum had no test positives.

## Retrieval

At K = 5 the classifier query gave Jaccard 0.558, nDCG 0.562, finding coverage 0.738, Hit 0.639 and a duplicate-text rate of 0.444; the oracle query reached a Jaccard of 0.820 and coverage of 0.974 (Table 4, Figure 3).

## Report generation

On the clinical subset finding precision was 0.413 (95% CI 0.342 to 0.489), recall 0.306 (95% CI 0.247 to 0.364), F1 0.351 (95% CI 0.295 to 0.407) and macro F1 0.233 (95% CI 0.165 to 0.304); hallucination rate 0.217 (95% CI 0.174 to 0.261) and omission rate 0.364 (95% CI 0.312 to 0.413); classifier false-positive propagation 0.356 (95% CI 0.290 to 0.419) and true-positive retention 0.702 (95% CI 0.612 to 0.788); ROUGE-L 0.220 (95% CI 0.211 to 0.230), BLEU-4 0.035 (95% CI 0.031 to 0.040) and the METEOR variant 0.204 (95% CI 0.192 to 0.217) (Table 5, Figures 5 and 6). All 486 generations succeeded in 4.5 s on average.

## Three-state system behaviour

Reference-normal studies were routed normal in 66.3%, abnormal in 23.3% and indeterminate in 10.4%; reference-abnormal studies in 23.5%, 65.1% and 11.4%. Among decided studies normal recall was 0.740 (95% CI 0.674 to 0.799) and abnormal recall 0.735 (95% CI 0.658 to 0.804), with decision coverage 0.891 (95% CI 0.859 to 0.921) (Table 6, Figure 7). Of the 40 abstained studies with a reference, 21 were reference-normal and 19 reference-abnormal; rare findings were not over-represented (Fisher p = 0.16). In 82 of 224 abnormal-routed studies (36.6%) the prose was read as normal (validation 31.0%).

## Grounding, copying and rare findings

Stated findings with retrieval-only provenance numbered 50 (16.0% reference-supported), against 54.5% of 132 findings supported by both sources. 33.3% of sentences were copied verbatim from the study's own Top-5 and 28.0% of reports copied a retrieved report entirely. Pooled retention of rare-finding classifier true positives was 6 of 22 (Table 7); counts are too small for strong claims.

## Validation to test and failure analysis

Test minus validation (Table 8, Figure 4): macro AUROC -0.017, macro AUPRC -0.091, Jaccard@5 +0.002, finding F1 -0.031, hallucination rate -0.039, decision coverage -0.045. Estimates outside the validation interval: macro AUPRC, ECE (macro), decision coverage. In the frozen failure taxonomy (Table 9, Figure 8), 142 of 368 studies (38.6%) fell in none of categories 1 to 8; the most frequent categories were classifier error (199), report-generator omission (134) and normal/abnormal classifier mismatch (86).
