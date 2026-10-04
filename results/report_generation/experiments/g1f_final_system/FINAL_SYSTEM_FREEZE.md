# Final system freeze (G1F)

**Freeze timestamp (UTC):** 2026-10-04T06:27:05+00:00

> Classification, calibration, operating thresholds, retrieval, query construction, Top-K, report-generation architecture, generator model, generation prompt and deterministic empty-output handling are frozen before opening the locked test set.

> No parameter, prompt, threshold, routing rule or model selection may be changed after inspecting the locked-test outputs.

## Identifiers

| Item | Value |
|---|---|
| Final configuration | `results/report_generation/experiments/g1f_final_system/FINAL_SYSTEM_CONFIG.json` |
| Configuration hash (SHA-256 of `settings`) | `cc04996cf8e8f6441505eedf5b41068b6af63384bf3a0a51bfde2a37c503eff5` |
| G1 prompt hash | `476b05326f85652b0cdb2faf79a4868a6e839660eb43e34d46cdae625e4b5688` |
| G1F guard hash (`src/system/guard.py`) | `1f0b340cc71cf261de84f4b19f290c99568a69760dc757d57c62d2c68c9397ee` |
| Classifier checkpoint SHA-256 | `6645ed44be7cb7d4cf4679b1fd561c7ac36be63f964ea320666199883126794b` |
| Calibration file SHA-256 | `50a14779157a60ff0f0ce973cf5b759801d67b17ec39f09949ee0056d7606c83` |
| R2 retrieval configuration SHA-256 | `9fb4d701a498de9f2e9a1ca325f645027ace0c72af0ef3a6180c0b85f34bc114` |
| Generator | `medgemma1.5:4b` digest `433252621ab154668b5d8be6aff6c1b771bacba045e46e6193da8d6ad1630f2c`, Q4_K_M, Ollama 0.35.1 |
| Locked retrieval-test study-id list SHA-256 | `2ca402e9ad3550d99240abf1f36fc1334d4fb45287b78e29cc3e24f0203b4780` (578 studies, opened: no) |

## Commits (recorded, not pushed)

C1: `a62073273cef155d6177d929a9d3a8e54279200b`, C2 results: `a2de4174b386ca8325c3481d626227dc205cd952`, C3: `031edd6306fd2974d9ae6107921ed92860416bf4`, C4: `77dee289fda740a1b261e31cc575343a49baea68`, C5: `55ba4bb87453a1dbb2a3452f5e6aebbcb57120b8`, C6: `15a276565eeb58d1e6f005833757921cc353f978`, R1: `f1a83bc396cc4ec126894b1641bc9317cbc73cf9`, R2: `0518b9aed626512a032aad52aacc4965878ff1ae`, G1: `9714f50d3f72587e91bf30cef238d4e5240f55e6`, G1A: `d4f78fca8601bb8a5146b307d4dcc74cd360d1f6`, G1B: `26dc0b34514d825710cf0d5a22bb59b33ae6cf43`, G2: `0f583d58466d39825060d781d41d97eedec0c419` (base head at the freeze, branch `g1f-final-system-guard`). The G1F files themselves are uncommitted at the time of the freeze.

## Selected architecture

G1 Single-Agent RAG (G2 is a negative-result experiment and is not part of the final system; see `FINAL_SYSTEM_SELECTION.md`).

## Frozen end-to-end pipeline

```
Chest X-ray
  -> preprocessing
  -> frozen DenseNet-121
  -> calibrated 14-label outputs
  -> C4 binary decisions
  -> deterministic routing guard (src/system/guard.py)

  Explicit No Finding (Path A, state normal)
    -> frozen G1 normal path

  >= 1 pathology positive (Path B, state abnormal)
    -> Top-3 classifier query findings
    -> Hybrid Dense + BM25 RRF
    -> Top-5 IU evidence reports
    -> frozen G1 MedGemma Single-Agent RAG
    -> preliminary Findings + Impression

  Empty classifier output (Path C, state indeterminate)
    -> deterministic INDETERMINATE output
    -> no retrieval
    -> no LLM generation
```

Routing details: Path A requires No Finding positive and no pathology label positive; Support Devices is not a pathology label, as in the frozen C4 rule (a device does not suppress No Finding). Path C requires that no label at all is positive and that No Finding is not positive. Path B is every other study, including a study whose only positive label is a support device without No Finding. The exact indeterminate output is:

`FINDINGS: Model output is indeterminate for this study.`

`IMPRESSION: Automated preliminary interpretation could not be established. Radiologist review is required.`

## Intended application behaviour for indeterminate studies

The interface must show **Automated interpretation unavailable — radiologist review required.** and must not show "No abnormality detected", "Normal X-ray" or any fabricated preliminary diagnosis for an indeterminate study. The displayed state comes from `system_interpretation_state` (normal, abnormal or indeterminate), never from the generated prose. The application is a research prototype, not an autonomous diagnostic system, and has not been built yet.

## Not included

No locked-test result of any kind. The retrieval and end-to-end test split is unopened.
