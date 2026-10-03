# Results: single-agent retrieval-augmented report generation (G1)

## Populations and generation

The primary set comprised 547 validation studies and the clinical-finding subset 378 (192 reference-normal); 547 studies had both a B0 and a G1 report (Table 1). The local model produced 547 of 547 reports without failure and 100.0% followed the requested format. Reports contained a mean of 23.9 words (B0 10.5).

## Clinical finding agreement

Finding precision was higher (an improvement) for G1 compared with B0 (0.411 (95% CI 0.354 to 0.469) versus 0.331 (95% CI 0.286 to 0.371)), recall was lower (worse) (0.358 (95% CI 0.300 to 0.416) versus 0.411 (95% CI 0.352 to 0.474)) and F1 did not differ reliably (0.383 (95% CI 0.330 to 0.435) versus 0.366 (95% CI 0.318 to 0.410); difference +0.017 (95% CI -0.024 to +0.053)) (Table 2, Figure 1). Counting only non-hedged assertions, the F1 difference was +0.019 (95% CI -0.021 to +0.059). Per-finding results are in Table 3.

## Hallucination and omission

25.7% of G1 reports and 40.7% of B0 reports contained at least one hallucinated finding (difference -0.151 (95% CI -0.193 to -0.116)), and 38.1% and 34.9% at least one omission (+0.032 (95% CI +0.005 to +0.061)); per report G1 stated 0.386 hallucinated and omitted 0.484 findings, against 0.627 and 0.444 for B0 (Table 4, Figure 2). Of the findings stated by G1, 2.8% were supported by the classifier only, 26.6% by retrieval only, 69.8% by both and 0.8% by neither (Figure 5). Findings stated by G1 that the classifier had not flagged (taken from the retrieved reports) numbered 68, of which 18 (26.5%) are in the reference.

## Classifier false-positive propagation

Classifier false-positive propagation was 0.405 (95% CI 0.350 to 0.462) for G1 and 1.000 (95% CI 1.000 to 1.000) for B0 (difference -0.595 (95% CI -0.650 to -0.538)), and true-positive retention 0.718 (95% CI 0.641 to 0.789) versus 1.000 (95% CI 1.000 to 1.000) (-0.282 (95% CI -0.359 to -0.211)) (Table 5, Figure 3). Yes, in this comparison G1 mentioned fewer classifier false positives than B0, at the price of retaining fewer classifier true positives (retention 0.718 (95% CI 0.641 to 0.789) versus 1.000 (95% CI 1.000 to 1.000)). B0 mentions every classifier-positive finding by construction (propagation and retention are both 1.000), so the informative quantity is how selectively G1 drops them: G1 kept 71.8% of the classifier true positives but 40.5% of the classifier false positives (difference +0.313). With definite assertions only the differences were -0.671 (95% CI -0.730 to -0.610) and -0.316 (95% CI -0.387 to -0.245). Retrieved support discriminated classifier errors only partly: mean support was 0.680 for true positives and 0.344 for false positives, and 85 of 237 false positives had no support. G1 left out 141 classifier false positives and 33 classifier true positives. This is a comparison of two complete systems: G1 differs from B0 in both the language model with its grounding prompt and the retrieved context, and no no-retrieval LLM ablation was run, so the effect cannot be attributed to retrieval alone.

## Normal and abnormal consistency

Against the reference, the classifier classified normal and abnormal studies with recall 0.729 and 0.726, B0 reports with 0.729 and 0.726, and G1 reports with 0.911 and 0.581; 0.0% of G1 reports and 6.3% of B0 reports were indeterminate.

## Copying, context use and lexical overlap

G1 copied 26.8% of its sentences of at least six words verbatim from the retrieved reports, and 21.9% of reports were whole-report copies. On average 4.253 of the five retrieved reports supported a stated finding, with the support spread over ranks 1 to 5 as 22.6%, 20.4%, 18.8%, 19.0%, 19.2%. ROUGE-L was 0.247 (95% CI 0.238 to 0.255) for G1 and 0.121 (95% CI 0.114 to 0.127) for B0 (difference +0.126 (95% CI +0.117 to +0.136)); BLEU-4 0.039 versus 0.010; the METEOR variant 0.227 versus 0.063 (Figure 4). BERTScore was not computed.

## Measurement validity and sensitivity

Applied to the reference report text, the finding extractor agreed with the MeSH-derived reference with precision 0.859, recall 0.877 and F1 0.868, which bounds the quality of every finding-level number above. When the same extractor was used for the reference (text-derived rather than MeSH-derived findings), the F1 was 0.390 for G1 and 0.375 for B0. Macro F1 over the 12 findings with reference positives was 0.215 for G1 and 0.325 for B0.

## Qualitative examples and error categories

Six deterministically selected cases are summarised in Table 6, and Figure 6 splits hallucinated findings into classifier false positives and other sources, and omissions into classifier true positives dropped by the report and findings never predicted by the classifier.

## Interpretation

These are validation-set results for one 4-billion-parameter local model with one frozen prompt. B0 and G1 differ in both the language model and the retrieved context, so the comparison does not isolate the contribution of retrieval. Finding extraction is rule-based and the reference is derived from MeSH terms. The locked retrieval-test partition has not been evaluated and no multi-agent system was tested.
