# University report content: G2 (structured multi-agent RAG)

_Insert-ready material generated from the G2 artifacts. Validation data only; classifier, retrieval configuration and generator model are unchanged; the locked retrieval-test split is unopened._

## Methodology

### Purpose and architecture

G2 splits report generation into three sequential roles of the same offline model (medgemma1.5:4b, greedy decoding, fixed seed): an Evidence Verifier that checks every classifier finding against the five retrieved reports and returns structured JSON, a Grounded Report Writer that sees only the classifier output and the verified evidence (never the retrieved text), and a Grounding Critic that checks the draft once and either approves it or returns one corrected report. There is no loop. The aim is to keep the recall that relevant retrieval gave the single-agent system G1 while reducing hallucination, classifier false-positive propagation, retrieval-induced unsupported findings and copying. The three prompts were frozen with SHA-256 hashes before bulk generation, and automated checks confirmed that no retrieved text reached the writer and the critic and that no reference text reached any agent.

### Evaluation

G2 is compared with G1 on the same validation studies with the same finding extractor, reference labels and metrics (finding precision, recall and F1, hallucination and omission, classifier false-positive propagation and true-positive retention, normal and abnormal recall, text-overlap scores) using a paired study-level bootstrap of 1,000 resamples. G2 is judged against predeclared criteria (lower propagation and hallucination with preserved recall, retention and abnormal recall, and less copying), not against a single metric.

## Results

### G1 versus G2

Finding F1 was 0.383 for G1 and 0.382 for G2 (difference -0.001 (95% CI -0.038 to +0.034)); recall +0.049 (95% CI +0.010 to +0.090); hallucination rate +0.124 (95% CI +0.087 to +0.164); classifier false-positive propagation +0.426 (95% CI +0.358 to +0.490); true-positive retention +0.214 (95% CI +0.138 to +0.287); abnormal recall +0.140 (95% CI +0.093 to +0.188). G2 met 4 of the six predeclared criteria relative to G1 (recall preserved (not reliably lower than G1); classifier TP retention preserved (not reliably lower than G1); abnormal recall preserved or improved (not reliably lower than G1); copying from the retrieved reports reduced) and did not meet: classifier FP propagation reduced relative to G1; hallucination rate reduced relative to G1. G2 is therefore not declared an improvement over G1 on this validation set. Criteria: classifier FP propagation reduced relative to G1: not met; hallucination rate reduced relative to G1: not met; recall preserved (not reliably lower than G1): met; classifier TP retention preserved (not reliably lower than G1): met; abnormal recall preserved or improved (not reliably lower than G1): met; copying from the retrieved reports reduced: met.

### Agents, retrieval-only findings and copying

The critic approved 56.3% of reports unchanged and modified 43.7% (239 of 547); in the clinical subset it removed 2 findings (2 not in the reference, 0 in the reference) and added 0 (0 in the reference, 0 not). Final G2 minus Agent 2 draft: F1 +0.001 (95% CI +0.000 to +0.003), recall +0.000 (95% CI +0.000 to +0.000), hallucination rate -0.003 (95% CI -0.008 to +0.000), FP propagation -0.008 (95% CI -0.022 to +0.000), TP retention +0.000 (95% CI +0.000 to +0.000). Agent 1 proposed 34 retrieval-only candidates, 10 reached the draft and 10 the final report. Copied-sentence rate against the study's own Top-5: G1 29.5%, G2 0.0% (-29.5 points (95% CI -33.3 to -26.0)); whole-report copies 24.1% and 0.0%. Pooled true-positive retention over the six rare findings: B0 1.000, G1 0.333 (7 of 21), G2 draft 0.810, G2 0.810 (17 of 21); other findings: G1 0.802, G2 0.958. G2 minus G1 retention is +0.476 (95% CI +0.263 to +0.722) for rare and +0.156 (95% CI +0.076 to +0.240) for other findings; rare-finding retention changed reliably.

### Runtime

G2 used 17.6 s per study against 3.7 s for G1 (4.8 times), with up to three sequential local calls per study; peak sampled GPU memory 4235 MiB.

## Discussion

Single-model, single-prompt-set, single-run validation result; the 4-billion-parameter model's critic and verifier are weak components, the agent roles share one model, and no result here shows clinical benefit. Additional measured limitations: Agent 1 support status did not separate correct from incorrect classifier findings; Agent 1 output was truncated for 10.4% of studies with context; the critic changed a stated finding in 2 studies; G2 reports are terse; empty classifier outputs were still written as normal reports. The decomposition is only as reliable as its weakest agent: the support status depends on the ranks cited by a small model, and the critic's own judgements are limited. The per-agent analyses show where the pipeline helps or does not. The multi-agent system is a research prototype, and the final application has not been built.
