# G1 representative validation examples

_Deterministic selection (first study by id satisfying each criterion; no manual choice, no editing of generated text). Identifiers are anonymised. IU 'truth' = MeSH-mapped findings of the study; reference text is shown for comparison only and was never part of the generator input._

## 1_correct_normal — case_4a7e5f5a46

- **Classifier:** no classifier-positive finding; No Finding positive
- **Query:** `no acute abnormality`; retrieved mapped findings: [1] No Finding; [2] No Finding; [3] No Finding; [4] No Finding; [5] No Finding
- **B0 (rule-based):** No acute cardiopulmonary abnormality is detected. No acute cardiopulmonary abnormality.
- **G1 generated report:** FINDINGS: The lungs and pleural spaces show no acute abnormality. Heart size and pulmonary vascularity within normal limits. IMPRESSION: 1. No acute pulmonary abnormality.
- **Reference report:** The cardiac and mediastinal silhouettes are unremarkable. The lungs are well expanded and clear. There are no focal air space opacities. There is no pneumothorax or effusion. There are mild degenerative changes of the thoracic spine. No evidence of acute cardiopulmonary process. Stable appearance of the chest.
- **Factual error analysis:** Reference (mapped) findings: none mapped. Classifier positives: none (false positives: none). G1 stated: none. Omitted reference findings: none. Retrieved mapped findings by rank: [1] No Finding; [2] No Finding; [3] No Finding; [4] No Finding; [5] No Finding.

## 2_correct_abnormal — case_179e5ffb9e

- **Classifier:** Enlarged Cardiomediastinum p=0.09 support 0/5; Cardiomegaly p=0.75 support 5/5; Lung Opacity p=0.61 support 3/5; Edema p=0.56 support 3/5; Atelectasis p=0.24 support 1/5; No Finding negative
- **Query:** `cardiomegaly lung opacity pulmonary edema`; retrieved mapped findings: [1] Atelectasis,Cardiomegaly,Lung Opacity; [2] Cardiomegaly,Edema,Lung Opacity,Pleural Effusion; [3] Cardiomegaly,Edema,Lung Opacity; [4] Cardiomegaly,Edema; [5] Cardiomegaly
- **B0 (rule-based):** Enlarged cardiomediastinal silhouette is detected. Cardiomegaly is detected. Pulmonary opacity is detected. Pulmonary edema is detected. Atelectasis is detected. Enlarged cardiomediastinal silhouette, cardiomegaly, pulmonary opacity, pulmonary edema and atelectasis.
- **G1 generated report:** FINDINGS: Cardiomegaly is present. There is a left lung base opacity consistent with atelectasis. IMPRESSION: Cardiomegaly and atelectasis.
- **Reference report:** Stable cardiac enlargement. Vascular congestion is redemonstrated. Patchy atelectasis and airspace disease is present in the left base. Right base is clear. No large effusion or pneumothorax.
- **Factual error analysis:** Reference (mapped) findings: ['Atelectasis', 'Cardiomegaly', 'Lung Opacity']. Classifier positives: ['Atelectasis', 'Cardiomegaly', 'Edema', 'Enlarged Cardiomediastinum', 'Lung Opacity'] (false positives: ['Edema', 'Enlarged Cardiomediastinum']). G1 stated: ['Atelectasis', 'Cardiomegaly', 'Lung Opacity']. Omitted reference findings: none. Retrieved mapped findings by rank: [1] Atelectasis,Cardiomegaly,Lung Opacity; [2] Cardiomegaly,Edema,Lung Opacity,Pleural Effusion; [3] Cardiomegaly,Edema,Lung Opacity; [4] Cardiomegaly,Edema; [5] Cardiomegaly.

## 3_classifier_fp_suppressed — case_9d07911f58

- **Classifier:** Lung Opacity p=0.53 support 4/5; Atelectasis p=0.21 support 1/5; Pneumothorax p=0.50 support 0/5; Pleural Effusion p=0.38 support 1/5; Pleural Other p=0.09 support 0/5; Fracture p=0.18 support 0/5; Support Devices p=0.49 support 0/5; No Finding negative
- **Query:** `lung opacity pneumothorax support device`; retrieved mapped findings: [1] Lung Opacity,Pleural Effusion; [2] Lung Opacity; [3] No Finding; [4] Lung Opacity; [5] Atelectasis,Lung Opacity
- **B0 (rule-based):** Pulmonary opacity is detected. Atelectasis is detected. Pneumothorax is detected. Pleural effusion is detected. Pleural thickening is detected. Fracture is detected. Support device is detected. Pulmonary opacity, atelectasis, pneumothorax, pleural effusion, pleural thickening, fracture and support device.
- **G1 generated report:** FINDINGS: Low lung volumes. Patchy opacity in the left lung base. No pneumothorax or pleural effusion. IMPRESSION: Patchy opacity in the left lung base, possibly atelectasis or early infiltrate.
- **Reference report:** The heart size and pulmonary vascularity appear within normal limits. The left hemidiaphragm remains elevated. Right base densities are again noted which appear improved. Previously seen left pleural effusion has resolved. There continues to be some left base opacities which may represent atelectasis. Surgical clips and suture lines are noted in the mediastinum. An air-fluid level is seen in the upper right abdomen immediately below the right hemidiaphragm. Extensive pleural densities are present on the right which may represent localized fluid or pleural thickening. No definite pneumothorax is seen. 1. Continued elevation of the left hemidiaphragm. 2. Extensive post-op changes. 3. Right pleural densities which may represent thickening or localized fluid. 4. Air-fluid level below the right hemidiaphragm medially. Although this could represent air-fluid within a of bowel, a subpulmonic process is a consideration. If clinically indicated, scan would better define this.
- **Factual error analysis:** Reference (mapped) findings: ['Lung Opacity', 'Support Devices']. Classifier positives: ['Atelectasis', 'Fracture', 'Lung Opacity', 'Pleural Effusion', 'Pleural Other', 'Pneumothorax', 'Support Devices'] (false positives: ['Atelectasis', 'Fracture', 'Pleural Effusion', 'Pleural Other', 'Pneumothorax']). G1 stated: ['Atelectasis', 'Lung Opacity']. Omitted reference findings: ['Support Devices']. Retrieved mapped findings by rank: [1] Lung Opacity,Pleural Effusion; [2] Lung Opacity; [3] No Finding; [4] Lung Opacity; [5] Atelectasis,Lung Opacity.

## 4_classifier_fp_propagated — case_c9e2d1042b

- **Classifier:** Enlarged Cardiomediastinum p=0.35 support 0/5; Lung Lesion p=0.16 support 0/5; Fracture p=0.21 support 2/5; No Finding negative
- **Query:** `enlarged cardiomediastinum lung lesion fracture`; retrieved mapped findings: [1] Atelectasis,Cardiomegaly; [2] no mapped finding; [3] Atelectasis,Fracture,Lung Opacity; [4] No Finding; [5] Cardiomegaly,Fracture
- **B0 (rule-based):** Enlarged cardiomediastinal silhouette is detected. Lung nodule or mass is detected. Fracture is detected. Enlarged cardiomediastinal silhouette, lung nodule or mass and fracture.
- **G1 generated report:** FINDINGS: Enlarged cardiomediastinal silhouette. Lung opacities noted at the lung bases. Fracture of the right clavicle. IMPRESSION: Cardiomegaly. Possible pneumonia and fracture.
- **Reference report:** Stable cardiomediastinal silhouette with mild cardiomegaly and aortic ectasia and tortuosity. No alveolar consolidation, no findings of pleural effusion. Chronic appearing bilateral rib contour deformities compatible with old fractures. No pneumothorax. No acute findings.
- **Factual error analysis:** Reference (mapped) findings: ['Cardiomegaly', 'Fracture']. Classifier positives: ['Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion'] (false positives: ['Enlarged Cardiomediastinum', 'Lung Lesion']). G1 stated: ['Cardiomegaly', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Opacity', 'Pneumonia']. Omitted reference findings: none. Retrieved mapped findings by rank: [1] Atelectasis,Cardiomegaly; [2] no mapped finding; [3] Atelectasis,Fracture,Lung Opacity; [4] No Finding; [5] Cardiomegaly,Fracture.

## 5_important_finding_omitted — case_50e4293b74

- **Classifier:** Cardiomegaly p=0.46 support 3/5; Lung Opacity p=0.50 support 4/5; Consolidation p=0.17 support 0/5; Pleural Effusion p=0.91 support 2/5; No Finding negative
- **Query:** `cardiomegaly lung opacity pleural effusion`; retrieved mapped findings: [1] Atelectasis,Cardiomegaly,Lung Opacity; [2] No Finding; [3] Cardiomegaly,Edema,Lung Opacity,Pleural Effusion; [4] Lung Opacity; [5] Atelectasis,Cardiomegaly,Lung Opacity,Pleural Effusion
- **B0 (rule-based):** Cardiomegaly is detected. Pulmonary opacity is detected. Consolidation is detected. Pleural effusion is detected. Cardiomegaly, pulmonary opacity, consolidation and pleural effusion.
- **G1 generated report:** FINDINGS: Cardiomegaly is present. There is opacity in the left lung base. IMPRESSION: 1. Cardiomegaly 2. opacity in the left base atelectasis
- **Reference report:** 1. No residual right pleural effusion identified status post thoracentesis. No pneumothorax. 2. Mediastinum, cardiac size grossly stable. 3. Small to moderate left-sided pleural effusion, increased compared to prior chest radiograph. Left lung base atelectasis/airspace disease.
- **Factual error analysis:** Reference (mapped) findings: ['Atelectasis', 'Lung Opacity', 'Pleural Effusion']. Classifier positives: ['Cardiomegaly', 'Consolidation', 'Lung Opacity', 'Pleural Effusion'] (false positives: ['Cardiomegaly', 'Consolidation']). G1 stated: ['Atelectasis', 'Cardiomegaly', 'Lung Opacity']. Omitted reference findings: ['Pleural Effusion']. Retrieved mapped findings by rank: [1] Atelectasis,Cardiomegaly,Lung Opacity; [2] No Finding; [3] Cardiomegaly,Edema,Lung Opacity,Pleural Effusion; [4] Lung Opacity; [5] Atelectasis,Cardiomegaly,Lung Opacity,Pleural Effusion.

## 6_retrieval_mismatch — case_db706fc12f

- **Classifier:** no classifier-positive finding; No Finding positive
- **Query:** `no acute abnormality`; retrieved mapped findings: [1] No Finding; [2] No Finding; [3] No Finding; [4] No Finding; [5] No Finding
- **B0 (rule-based):** No acute cardiopulmonary abnormality is detected. No acute cardiopulmonary abnormality.
- **G1 generated report:** FINDINGS: The lungs and pleural spaces are clear. The heart size is normal. The pulmonary vascularity is normal. IMPRESSION: No acute pulmonary abnormality.
- **Reference report:** Cardiac and mediastinal contours are within normal limits. The lungs are clear. Left axillary surgical clips. Bony structures are intact. No active pulmonary disease.
- **Factual error analysis:** Reference (mapped) findings: ['Support Devices']. Classifier positives: none (false positives: none). G1 stated: none. Omitted reference findings: ['Support Devices']. Retrieved mapped findings by rank: [1] No Finding; [2] No Finding; [3] No Finding; [4] No Finding; [5] No Finding.
