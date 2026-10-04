| Component | Specification |
|---|---:|
| Pipeline | Agent 1 (Evidence Verifier) -> deterministic validation -> Agent 2 (Grounded Report Writer) -> Agent 3 (Grounding Critic); one pass, no loop |
| Frozen upstream components | DenseNet-121 classifier, C4 operating policy, No Finding rule, C5 Platt calibration, R2 retrieval (top-3 finding query, dense + BM25 RRF, Top-5) |
| Generator (all agents) | medgemma1.5:4b (Ollama 0.35.1, digest 433252621ab1, Q4_K_M), local only, no tools, no web |
| Decoding | temperature 0.0, top_k 1, top_p 1.0, seed 42, context 8192 tokens |
| Output budget (new tokens) | agent1 900, agent2 400, agent3 700 (all agents use schema-constrained JSON output) |
| Agent 1 input | classifier-positive findings, calibrated probabilities, No Finding state, frozen Top-5 retrieved reports (rank and text only) |
| Agent 1 output | JSON: supporting ranks, short evidence summary and support status per classifier finding; retrieval_only candidates; ranks of normal-describing reports |
| Deterministic validation | valid ranks only; support count recomputed from ranks; status from count (>=2 supported, 1 partially supported, 0 unsupported); summaries reproducing retrieved prose replaced |
| Agent 2 input | classifier state, calibrated probabilities, validated Agent 1 evidence JSON; no retrieved report text |
| Agent 3 input | classifier state, validated Agent 1 evidence JSON, Agent 2 draft; no retrieved report text; APPROVE keeps the draft unchanged, otherwise one corrected report |
| Prompt hash Agent 1 | ccca211d17867114afc3eff6467db3e1dfe2a9df83c5697c0e5a39978af51038 |
| Prompt hash Agent 2 | a41126c192a59c4dc1f62df218a1dc4bf3ba883d28df5883895c01f539e78b99 |
| Prompt hash Agent 3 | 3be25ca4a0e93091b17643a696a1201851ee181ff35b88814ecd2f4924fcef46 |
| Combined prompt hash | 885a6793bec33a1c308c85fc682c457bd5cb380ef7192ffd463cdc61eff2e80c |
