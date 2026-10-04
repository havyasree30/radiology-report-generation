# F1 locked end-to-end test protocol

_Registered 2026-10-04T06:38:34+00:00 UTC on branch `f1-locked-end-to-end-test` at commit `558f77281c100f1e0671a315cd6e156478534fb5`; written BEFORE any locked-test report content, image, classifier output or result was read. The JSON version is authoritative; its SHA-256 is `dee43410b37b3eaaa7f810d80e0e7bc776ba92f2a885702c350c93c7c4a48a08`._

- Final system configuration SHA-256: `cc04996cf8e8f6441505eedf5b41068b6af63384bf3a0a51bfde2a37c503eff5`
- G1 prompt SHA-256: `476b05326f85652b0cdb2faf79a4868a6e839660eb43e34d46cdae625e4b5688`
- Guard SHA-256: `1f0b340cc71cf261de84f4b19f290c99568a69760dc757d57c62d2c68c9397ee`
- Locked-test study-ID list SHA-256: `2ca402e9ad3550d99240abf1f36fc1334d4fb45287b78e29cc3e24f0203b4780` (578 ids; opened before this protocol: no)

## Statement

> No model, threshold, calibration, query, retrieval setting, Top-K, generator, prompt, routing rule, verifier or postprocessor may be changed after any locked-test reference content or metric is inspected; no study may be excluded because of poor performance; any change would require a new independent test set.

## Evaluation populations

- **P0 locked studies**: all 578 locked test studies (report uid is the unit)
- **P1 with frontal image**: studies with at least one Frontal image; the frontal image is the first Frontal file in sorted filename order (R1 rule)
- **P2 classified**: P1 studies whose image passes the frozen ImageValidator and is classified; studies failing the validator are listed with the reason, never silently dropped
- **P3 primary end to end set**: P2 studies with a non-empty cleaned reference report (build_retrieval_text) = the G1 'primary report-generation set'; routing counts are reported on P3 and on P2
- **P4 clinical finding subset**: P3 studies with a usable mapped truth (has_eval_set under the frozen r1-v1 MeSH mapping); all finding-level, report-state and abstention metrics use P4
- **classification population**: P2 studies with a usable mapped truth (R1 'eligible and classified'); label = class in the mapped truth set, No Finding = truth is exactly {No Finding}; unmapped MeSH terms are NOT verified negatives (precision is a lower bound)
- **retrieval population**: P2 studies with a usable mapped truth and a non-empty classifier query (R1/R2 'primary set')
- **inclusion rules changed after seeing test**: false
- **reported without silent exclusion**: ["total locked studies", "with frontal image", "classified", "with usable reference report", "with usable mapped truth (P4)", "routed to Path A / B / C (on P3 and on P2)", "excluded studies with reasons"]

## Classification

- **definitions**: identical to C6: scripts/c6_03_evaluate.evaluate (per-class AUROC/AUPRC/precision/recall/specificity/F1/balanced accuracy, macro and micro averages over defined classes, Brier, log loss, adaptive-bin ECE with 10 bins)
- **metrics**: ["macro AUROC", "micro AUROC", "macro AUPRC", "micro AUPRC", "macro precision", "macro recall", "macro specificity", "macro F1", "macro balanced accuracy", "Brier score", "log loss", "ECE"]
- **per class**: ["AUROC", "AUPRC", "precision", "recall", "F1", "support"]
- **undefined classes**: reported as undefined (no positives or no negatives), never replaced by 0
- **saved**: ["raw sigmoid outputs", "calibrated probabilities", "binary predictions after the No Finding rule", "No Finding rule result", "routing state"]
- **preprocessing and inference**: frozen: fp32_strict, no TTA, no ensembling, image size 224

## Routing

- **rule**: src/system/guard.route (frozen); state never derived from prose
- **states**: ["normal", "abnormal", "indeterminate"]

## Retrieval

- **configuration**: frozen R2 (hybrid RRF, Top-5 for generation, Top-10 retained for the K analysis)
- **K**: [1, 3, 5, 10]
- **primary K**: 5
- **metrics**: ["Jaccard (truth)", "nDCG", "union finding coverage", "Hit and reciprocal rank (exact finding-set match)", "duplicate-text rate", "zero-overlap rate"]
- **definitions**: src/retrieval/evaluation.evaluate_ranking (unchanged)
- **oracle diagnostic**: oracle query (the study's true findings) through the same hybrid pipeline on the same population; classifier-query minus oracle gap at K=5; diagnostic only, may not alter the system
- **path C**: no retrieval
- **path A normal query**: no acute abnormality (frozen R2 phrase)

## Generation

- **system**: frozen G1 (prompt hash above); Paths A and B each generated exactly once; Path C returns the frozen indeterminate text without any call
- **model options**: {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "seed": 42, "num_predict": 400, "num_ctx": 8192, "stop": ["A. AUTOMATED CLASSIFIER OUTPUT", "B. RETRIEVED EVIDENCE", "C. EVIDENCE SUMMARY"]}
- **cache**: every response cached per study with the request hash and model digest
- **retry policy**: generate_single: up to 5 attempts for transient errors with exponential back-off (2^attempt s, max 60 s); malformed requests are not retried
- **persistent failure handling**: recorded explicitly and listed; never replaced by fabricated or hand-edited text; report metrics are computed on studies with a generated report and the failures are reported as a count with ids; no re-prompting

## Report metrics

- **set**: P4
- **metrics**: ["finding precision", "finding recall", "finding F1 (micro)", "macro finding F1", "hallucination rate", "omission rate", "classifier FP propagation", "classifier TP retention", "ROUGE-L", "BLEU-4", "METEOR exact-match variant (not standard METEOR)"]
- **text metric set**: P3 (all studies with a final text, including the indeterminate message)
- **definitions**: src/generation/study_eval.study_row (unchanged); finding extractor = frozen Phase 1 lexicon + finding_extraction_changes.yaml; 'stated' = affirmative or hedged mention

## Three state

- **set**: P4
- **outputs**: ["state distribution (P3 and P4)", "routing by reference state", "normal and abnormal recall among decided studies", "decision coverage = decided / eligible (P4)"]
- **indeterminate not forced into normal or abnormal**: true

## Abstention

- **descriptive**: true
- **outputs**: ["number and share", "reference-normal and reference-abnormal fractions", "findings in reference-abnormal abstentions", "rare-finding representation (Fisher exact, descriptive)"]

## Routing prose consistency

- **definition**: among studies routed abnormal, the share whose final report prose is read by the frozen extractor as normal / abnormal / indeterminate; abnormal-routing -> normal-prose rate; reports are not repaired; the validation rate (31.0%) is a descriptive comparator and not a target or threshold

## Provenance

- **classes**: ["A classifier only", "B retrieval only", "C both", "D neither"]
- **definition**: src/generation/metrics.provenance with the study's own Top-5 mapped findings as the retrieval-supported set
- **reported**: ["counts", "reference support within each class", "retrieval-only supported vs unsupported"]

## Copying

- **definition**: src/generation/metrics.copy_stats (min 6 words) against the study's own Top-5 text + repeated-sentence rate; unchanged
- **reported**: ["sentence copying", "whole-report copying", "repeated-sentence rate"]

## Rare findings

- **findings**: ["Pleural Other", "Pneumonia", "Fracture", "Lung Lesion", "Enlarged Cardiomediastinum", "Consolidation"]
- **reported**: ["test support", "precision", "recall", "F1", "TP retention"]
- **caution**: no strong claim when counts are very small

## Bootstrap

- **resamples**: 1000
- **seed**: 42
- **unit**: study (IU X-Ray has no patient identifier; one frontal image per study); patient-level resampling is not possible for this population
- **interval**: 95% percentile
- **significance testing**: none beyond descriptive intervals and the descriptive Fisher test above

## Validation to test

- **method**: the same analysis code is run on the validation partition (dry run; stored G1/G1F/R1/R2 results must be reproduced) and test-minus-validation absolute differences are reported
- **classification comparator**: IU validation (same pipeline and labels); CheXpert C6 validation/test values are listed as context only
- **use**: descriptive; differences are not used for tuning

## Analysis code

- **development**: all F1 analysis code is developed and debugged on the VALIDATION partition only (dry run) and must reproduce the stored validation results before the test is opened; the SHA-256 of every F1 script is recorded in f1_code_freeze.json immediately before the first locked-test access
- **bug rule**: a code bug found after the test is opened is documented with the corrected code hash and may be fixed only if it invalidates a metric; values are never edited and no scientific choice is changed

## Frozen failure taxonomy

- **1_classifier_error**: classifier non-No-Finding positive set differs from the mapped truth findings
- **2_normal_abnormal_classifier_mismatch**: classifier state (normal/abnormal) differs from the reference state (indeterminate studies are category 8)
- **3_retrieval_query_mismatch**: studies with retrieved context whose Top-5 union contains none of the reference findings (union coverage@5 = 0)
- **4_retrieval_only_unsupported_finding**: the report states a finding with provenance 'retrieval only' that is not in the reference
- **5_report_generator_omission**: the report omits at least one reference finding
- **6_report_generator_hallucination**: the report states at least one finding that is not in the reference
- **7_abnormal_routing_normal_prose**: routed abnormal but the final prose is read as normal
- **8_indeterminate_abstention**: routed indeterminate
- **9_corpus_limitation**: no corpus report has exactly the reference finding set (exact match impossible)
- **10_reference_label_limitation**: the study has unmapped or uncertain MeSH terms (reference possibly incomplete)

## Planned figures

- 1 per-class classifier AUROC/AUPRC
- 2 calibration / reliability
- 3 retrieval K analysis
- 4 validation vs test classification metrics
- 5 report precision/recall/F1
- 6 hallucination, omission, FP propagation, TP retention
- 7 three-state results
- 8 end-to-end failure-category summary

## Planned tables

- 1 locked-test population
- 2 final classifier metrics
- 3 per-class classifier metrics
- 4 final retrieval metrics
- 5 final report-generation metrics
- 6 three-state system metrics
- 7 rare-finding analysis
- 8 validation-to-test comparison
- 9 failure analysis
- 10 runtime/compute

## Runtime

- classifier inference runtime
- retrieval runtime
- G1 generation runtime (per call and total)
- total runtime
- Ollama/model metadata
- sampled peak GPU memory (nvidia-smi)
