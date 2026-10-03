# Results: retrieval baseline and vision-to-retrieval evaluation (R1)

## Dataset, split and mapping

The collection contained 3,851 studies and 7,466 images. The reference corpus comprised 2,675 studies and the retrieval-validation partition 578; no study was shared between partitions and no retrieved report was the query study (Table 1). Of the validation studies, 403 had a usable finding set (207 normal), 382 of them had a frontal image, and 358 had a non-empty classifier query and formed the primary set. Embeddings had dimension 384.

## Oracle retrieval and the retriever comparison

With oracle finding queries, dense retrieval reached a Jaccard@3 of 0.805 (95% CI 0.778 to 0.832) and nDCG@3 of 0.809; lexical retrieval reached 0.787 (95% CI 0.755 to 0.816) and 0.793. Random retrieval would give about 0.213. The paired dense-minus-lexical difference was +0.018 (95% CI -0.002 to +0.038), so the two retrievers were not reliably different (Table 2, Figure 1). Hit@3 was 0.906 for dense and 0.818 for lexical retrieval.

## Frozen-classifier queries

Using queries from the frozen classifier lowered agreement substantially: Jaccard@3 was 0.514 (95% CI 0.465 to 0.560) (dense) and 0.533 (95% CI 0.484 to 0.577) (lexical), which is 0.291 and 0.254 below the oracle (Table 3, Figure 2). With these queries lexical retrieval was slightly better than dense retrieval (paired Jaccard@3 difference -0.019, 95% CI -0.035 to -0.003). Retrieval followed the query: the Jaccard@3 against the query's own findings was 0.691 (dense), higher than against the truth. Among the 200 dense top-1 failures, 190 arose from a classifier query that differed from the true finding set and 10 from retrieval despite a correct query; the commonest subtype was normal-versus-abnormal disagreement (77).

## Precision-aware gating

Gates were derived on CheXpert validation data for 12 of 13 findings (Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Fracture, Support Devices); Pleural Other received no gate. They raised class-level precision at a cost in recall (Table 6) and removed 251 of 448 frozen-positive finding instances from the IU validation studies, shortening queries from 1.50 to 1.16 findings. 18.7% of studies then used the strongest-positive fallback. No consistent improvement was found: Jaccard@3 differed by +0.006 (95% CI -0.008 to +0.021) (dense) and -0.003 (95% CI -0.015 to +0.010) (lexical), and nDCG@3 by +0.006 (95% CI -0.009 to +0.021) and -0.008 (95% CI -0.020 to +0.006). For lexical Jaccard@1 the gated policy was lower: -0.024 (95% CI -0.044 to -0.003). Precision-aware gating therefore did not improve retrieval over passing all frozen positives, and it does not alter classifier performance.

## Performance across K

Jaccard@K varied by at most 0.038 between K = 1 and K = 10 for any policy and retriever (Table 4, Figure 3). Union coverage of the true findings with oracle dense retrieval increased from 0.855 (K=1) to 0.964 (K=3) and 0.977 (K=10), whereas with all-positive classifier queries it increased from 0.574 to 0.646 and 0.814. Duplicate-text rates rose with K (34.8% at K=3 and 47.3% at K=10 for dense oracle retrieval). The per-query distribution of top-3 agreement was bimodal for classifier queries, with 37.9% of retrieved reports sharing no finding (Figure 4).

## Classifier behaviour on IU X-Ray

The frozen classifier produced no output for 49 of 551 studies (8.9%) and predicted No Finding for 48.8% against 51.3% indexed normal. It over-predicted, for example, Fracture (17.2% predicted versus 2.4% in the mapped truth), and under-predicted Support Devices (2.5% versus 9.2%). Macro precision, recall and F1 over the 13 abnormal findings were 0.313, 0.449 and 0.297 (descriptive only; Table 5, Figure 5).

## Interpretation

Both retrievers recover the true findings well when the query is correct, and most of the loss in the end-to-end pipeline arises before retrieval. Top-3 is a defensible default for oracle-quality queries (most of the achievable coverage, no Jaccard loss) but it is not shown to be optimal: for classifier-driven queries coverage still improves at larger K, redundancy grows with K while the share of retrieved reports without finding overlap stays roughly constant, and the choice of K is deferred to R2 where it can be judged on generated reports. The locked retrieval-test partition has not been evaluated.
