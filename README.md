# PD NeuroCAPs — Co-Activation Pattern Analysis of Parkinson's Disease fMRI

This repository contains the complete analysis pipeline and figure generation code for the manuscript:

> **"Dynamic depletion of a cortico–basal ganglia–cerebellar network state is associated with dopaminergic treatment resistance in Parkinson’s disease"**

The project investigates large-scale brain network dynamics in Parkinson's disease (PD) using **Co-Activation Pattern (CAP)** analysis on resting-state functional MRI (rs-fMRI) data, comparing PD patients (both OFF and ON medication) against healthy controls (HC).

---

## Overview

![Figure1](figure/Figure1.png)

CAP analysis identifies recurrent whole-brain co-activation states from concatenated fMRI time series via K-means clustering. This pipeline covers the full workflow:

1. **Timeseries extraction** from BIDS-formatted rs-fMRI data using a custom 283-region atlas
2. **K-means clustering** to identify 7 CAPs (validated via elbow criterion and bootstrap stability)
3. **CAP metric computation** — counts, persistence (mean dwell time), temporal fraction, transition frequency, and transition probability
4. **Cross-sectional comparison** (HC vs PD OFF) using ANCOVA
5. **Longitudinal comparison** (PD OFF vs ON medication) using Linear Mixed Models (LMM)
6. **Clinical correlation** — partial Spearman correlations between CAP metrics and UPDRS-III scores
7. **Treatment response prediction** — SVM classification of responders vs non-responders
8. **Gender-stratified analysis** — three-way LMM interaction (CAP × Sex × Condition)

---

## Requirements

The pipeline is built with **Python 3.14+** and depends primarily on the following packages:

| Package | Purpose |
|---|---|
| `neurocaps` | CAP analysis (timeseries extraction, clustering, metrics) |
| `nilearn` | Neuroimaging data processing & surface plotting |
| `nibabel` | NIfTI image I/O |
| `numpy`, `pandas`, `scipy` | Scientific computing |
| `statsmodels` | OLS / MixedLM / FDR correction |
| `pingouin` | Partial Spearman correlation |
| `scikit-learn` | KMeans, SVM, pipeline, metrics |
| `matplotlib`, `seaborn` | Publication-quality figures |
| `ptitprince` | Raincloud plots |
| `tqdm` | Progress bars |

Install with:

```bash
pip install -r requirements.txt
```

> **Note**: The `neurocaps` package and its dependencies may require specific system libraries for NIfTI I/O. See the [neurocaps](https://github.com/donishadsmith/neurocaps) for detailed installation instructions.

---

## Data

### Input Data Requirements

- **BIDS-formatted resting-state fMRI dataset** with preprocessed images (the raw data is not included in this repository)
- **Clinical CSV** (`data/merged_clinical_metrics.csv`) containing:
  - Demographics: Age, Sex, Years of Education
  - Clinical scores: UPDRS-III (OFF and ON medication), H-Y stage, LEDD
  - Group labels: HC (healthy control) or PD (Parkinson's disease)
  - Responder status (binary, based on ≥30% UPDRS-III improvement)

### Custom Atlas

The analysis uses a **283-region cortical/subcortical atlas** combining:
- **BN234** — Brainnetome Atlas (234 cortical/subcortical regions)
- **CIT18** — Cerebellar atlas (18 regions)
- **SUIT34** — SUIT cerebellar atlas (34 regions, de-duplicated against CIT)

Defined in `atlas283_custom_parcellation.py` with the corresponding NIfTI file at `data/atlas/Atlas283_combined_BN234_CIT18_SUIT34_2mm.nii`.

---

## Pipeline Execution

### Step 1 — Timeseries Extraction & CAP Identification

Run the Jupyter notebook:

```bash
jupyter notebook PDNeuroCAPs.ipynb
```

The notebook performs:
1. BIDS dataset extraction using `neurocaps.TimeseriesExtractor`
2. K-means clustering (k=7) on concatenated subject×run time series
3. CAP metric computation (counts, persistence, temporal fraction, transition frequency/probability)
4. Bootstrap clustering stability evaluation (optional, via `scripts/bootstrap_clustering_stability_utils.py`)

![Figure2](figure/Figure2.png)

### Step 2 — Statistical Analyses

Run the statistical analysis scripts in order:

```bash
# Demographics & LEDD comparison
python clinical_statistics_LEDD.py

# Framewise displacement (head motion) comparison
python FD_statistics.py
```

The main statistical pipeline (invoked from the notebook or independently):

| Script | Analysis | Design |
|---|---|---|
| `statistics_ancova_lmm_utils.py` | Cross-sectional ANCOVA | `CAP ~ Group + Sex + Age + Edu + FD` |
| `statistics_ancova_lmm_utils.py` | Longitudinal LMM | `CAP ~ Condition + Sex + Age + Edu + FD + (1\|Subject)` |
| `statistics_correlation_utils.py` | Partial correlation | CAP metrics ↔ UPDRS-III (controlling for covariates) |
| `response_lmm_interaction_utils.py` | LMM interaction | `CAP ~ Cond × RespGroup + covariates + (1\|Subject)` |
| `response_llm_stratified_utils.py` | Three-way LMM | `CAP ~ Cond × RespGroup × Sex + covariates + (1\|Subject)` |
| `svm_response_prediction_utils.py` | SVM classification | Predict responder vs non-responder from CAP features |

All p-values are corrected for multiple comparisons using **Benjamini-Hochberg FDR** within each metric family.

---

## Output Structure

| Directory | Contents |
|---|---|
| `outputs/PDNeuroCAPs/01_extraction/` | Extracted ROI time series (CSV + NPZ) and QC report |
| `outputs/PDNeuroCAPs/02_clustering/` | K-means models, cluster labels |
| `outputs/PDNeuroCAPs/03_metrics/` | CAP-derived metric tables (long format + per-group) |
| `outputs/PDNeuroCAPs/04_framewise_labels/` | Frame-by-frame CAP assignments |
| `outputs/PDNeuroCAPs/05_cap_templates/` | CAP spatial templates and radar cosine similarity |
| `outputs/PDNeuroCAPs/06_statistics_datasets/` | Merged cross-sectional and longitudinal wide-format tables |
| `outputs/PDNeuroCAPs/07_statistics_ANCOVA_LMM/` | ANCOVA and LMM result tables |
| `outputs/PDNeuroCAPs/08_statistics_correlation/` | Partial correlation results |
| `outputs/PDNeuroCAPs/09_response_lmm_interaction/` | LMM interaction results |
| `outputs/PDNeuroCAPs/10_response_lmm_stratified/` | Stratified three-way LMM results |
| `outputs/PDNeuroCAPs/11_svm_response_prediction/` | SVM predictions, bootstrap NRI comparison |
| `outputs/clinical_statistics/` | Clinical demographics and LEDD/H-Y comparison tables |

---

## Data Availability Note

- Please note that the raw MRI imaging data (BIDS-formatted resting-state fMRI dataset) and the extracted ROI time series data (`01_extraction`) are not included in this repository due to patient privacy, ethical confidentiality constraints, and file size limitations.
- To obtain these specific datasets for reproduction, please contact the corresponding authors (Tao Feng or Tao Liu) upon reasonable request. 
- All other derived metrics and machine learning features are fully provided. 

---

## Citation

If you use this code in your research, please cite the associated manuscript (citation information will be added upon publication).

---

## License

This repository is provided for research and reproducibility purposes. Please contact the corresponding author for any inquiries regarding data access or usage.
