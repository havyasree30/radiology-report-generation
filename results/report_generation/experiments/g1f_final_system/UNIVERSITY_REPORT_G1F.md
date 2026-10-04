# University report content: G1F (final system guard and freeze)

_Insert-ready material generated from the G1F artifacts. Validation data only; the locked retrieval/end-to-end test is unopened._

## Methodology

### Final report generator

Two report generators were compared on the validation set: the single-agent retrieval-augmented generator G1 and a three-agent variant G2. G1 was selected: finding F1 was 0.383 for G1 and 0.382 for G2, but G2 had lower precision (0.359 versus 0.411), higher classifier false-positive propagation (0.831 versus 0.405), a higher hallucination rate (0.381 versus 0.257) and about 4.8 times the runtime. Its gains in recall and abnormal recall came largely from repeating classifier-positive findings, its verifier did not separate supported from unsupported findings and its critic changed almost nothing.

### Deterministic routing guard

The classifier produces 49 empty outputs on the validation set (no positive label and No Finding not positive). Earlier systems wrote a normal-looking report for them. A deterministic guard now routes every study to one of three states: normal (No Finding positive and no pathology label positive; frozen G1 normal path), abnormal (any other study with a positive label; frozen G1 retrieval-augmented report) and indeterminate (empty output; no retrieval, no language-model call, and the fixed message "Model output is indeterminate for this study. Automated preliminary interpretation could not be established. Radiologist review is required."). The state comes from the classifier output, not from the generated text. Existing G1 reports were reused unchanged, so no new generation was performed.

### Freeze

Classification (checkpoint, thresholds, No Finding rule, calibration), retrieval (query construction, hybrid reciprocal-rank fusion, Top-5), the generator model and prompt and the guard were recorded in one configuration with SHA-256 `cc04996cf8e8f6441505eedf5b41068b6af63384bf3a0a51bfde2a37c503eff5` before the locked test set is opened. No parameter, prompt, threshold, routing rule or model selection may be changed after inspecting locked-test outputs.

## Results

Routing: 269 normal, 229 abnormal and 49 indeterminate studies (9.0%). Finding-level validation metrics are identical to the original G1 (F1 0.383, precision 0.411, recall 0.358, hallucination rate 0.257, false-positive propagation 0.405, true-positive retention 0.718); text-overlap scores fall slightly (ROUGE-L 0.247 to 0.225). Among decided studies, normal recall was 0.781 and abnormal recall 0.760, with decision coverage 93.7%. Of the 24 abstained studies with a reference, 15 were reference-abnormal and 9 reference-normal; rare findings were not over-represented (descriptive, small counts).

## Discussion

The guard turns an unrecognised failure (a normal-looking report for an empty classifier output) into an explicit abstention at the price of 6.3% lower coverage in the clinical subset. It does not improve the classifier or the generator: the reused G1 prose still reads normal for 31.0% of abnormal-routed studies, so the interface must display the routing state and the indeterminate message, never a normal label for an indeterminate study. The system remains a research prototype and the final application has not been built.
