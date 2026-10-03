# Methods: sham-retrieval control (G1B)

## Purpose and design

The single-agent retrieval-augmented system (G1) writes a preliminary report from the frozen classifier output and the Top-5 retrieved corpus reports. To test whether the relevance of the retrieved content matters, and not merely the presence of radiology-report text, we ran a sham-retrieval control (G1B). G1B used the same 547 validation studies, the identical G1 prompt, the identical local model medgemma1.5:4b (Ollama 0.35.1, digest verified to equal that of G1), greedy decoding with temperature 0, top-k 1, seed 42, a 8192-token context and at most 400 new tokens, the same message format and number of context reports, the same classifier outputs, the same report parser, finding extractor, reference labels and metrics. Only the identity of the five context reports differed.

## Sham context

For each of the 498 studies with a retrieval query, five reports were drawn without replacement from the 2675 training reports of the retrieval corpus, with a per-study generator seeded from a fixed seed (42) and the study identifier only. Selection never used the reference report, reference findings, classifier output or image. The query study and the study's own true Top-5 were excluded, no locked-test report was eligible, and the empty-query studies were handled exactly as in G1. Rank slots and the displayed rank fields were preserved, and the generator was not told that the context was a control. The evidence-support counts were computed by the unchanged G1 function from the context supplied. The mapping was frozen with a hash before generation and not changed afterwards.

## Comparisons and outcomes

Four systems were compared on identical studies: the rule-based baseline (B0), the no-retrieval ablation (G1A, a classifier-only robustness check), G1B and G1. The primary comparison was G1 minus G1B. Outcomes were finding precision, recall, micro and macro F1, hallucination and omission rates, propagation of classifier false positives, retention of classifier true positives, normal and abnormal recall of the report state, ROUGE-L, BLEU-4, an exact-match METEOR variant and report length, computed as in G1 on the primary set (547 studies) and the clinical-finding subset (378 studies). Uncertainty was a paired study-level bootstrap with 1,000 resamples and the seed used in G1; a difference was interpreted as an effect of context relevance only when its interval excluded zero.

## Additional analyses

Findings introduced beyond the classifier output (stated, not flagged by the classifier) and classifier positives removed were counted per system and checked against the reference. Findings stated under one condition only were compared finding by finding. Copying was measured against each system's own context reports (sentences copied verbatim, whole-report copies), against the other condition's context as a control, and against all corpus reports (sentences of at least six words, template sentences, whole-report duplication). Retention of true positives was analysed for the six rare findings of the loss study and for all others, and normal and abnormal behaviour as the report state against the reference and as the share of normal reports given an abnormal classifier output. A context check compared how closely the relevant and sham contexts matched the reference findings.
