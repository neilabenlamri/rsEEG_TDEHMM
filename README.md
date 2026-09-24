# Resting-state EEG TDE-HMM analysis in acute stroke

This repository contains the analysis code used to study transient, state-resolved
resting-state EEG dynamics in acute stroke with a time-delay embedded hidden
Markov model (TDE-HMM).

The central aim is to determine whether dynamic EEG features derived from
recurrent brain states provide information about motor impairment and recovery
beyond conventional static spectral power.

## Research questions

The analysis addresses four main questions:

1. Do healthy participants and patients with acute stroke differ in the
   temporal expression of TDE-HMM states?
2. Which spectral and spatial power characteristics distinguish the groups?
3. Do state-resolved dynamic features provide information that is not visible
   in static, time-averaged EEG power?
4. Do regional dynamic EEG features improve prediction of follow-up motor
   function beyond baseline clinical information and static power?

## Analysis overview

The workflow:

1. preprocesses and standardizes source-space resting-state EEG;
2. applies time-delay embedding and principal-component reduction;
3. trains an eight-state TDE-HMM;
4. estimates trial-level state probability time courses;
5. derives subject-level temporal features, including fractional occupancy and
   switching rate;
6. estimates state-resolved power spectral density and coherence;
7. compares healthy and stroke groups using subject-level statistical tests;
8. maps static and dynamic power onto the DKT cortical parcellation;
9. relates regional EEG features to Fugl–Meyer motor scores at T1, T2 and T3;
10. evaluates static and dynamic EEG features in nested cross-validated ridge
    models of follow-up motor outcome.

Trials are summarized within participants before group-level inference. The
participant is therefore the independent observational unit.

## Static and dynamic EEG features

Static power is obtained by combining state-specific spectra using each
participant's fractional occupancy. It represents the participant's
time-averaged spectral organization.

Dynamic power describes state-specific spectral deviations relative to a
common static reference. Dynamic values may therefore be positive or negative:
their sign describes relative state expression and does not imply negative
absolute physiological power.

The principal comparison asks whether these state-resolved dynamic features
add predictive information beyond static power and baseline clinical
variables.

## Statistical inference

The notebook implements subject-level inference, including:

- two-sided label-permutation tests for fractional occupancy and switching rate;
- Welch tests of log-transformed absolute band power;
- cluster-mass label-permutation tests across adjacent frequency bins;
- parcel-wise Welch tests for static and dynamic spatial power;
- Benjamini–Hochberg false-discovery-rate control over the explicitly defined
  state, band, parcel or edge families;
- Hedges' *g* effect sizes, reported as stroke minus healthy;
- partial Spearman correlations between regional EEG power and raw
  Fugl–Meyer scores, adjusted for age, categorical education and lesion size;
- nested repeated cross-validation for ridge-regression prediction of FM-T2
  and FM-T3 from baseline FM-T1, clinical covariates and EEG summaries.

The prediction figures report out-of-fold \(R^2\), RMSE and the incremental
out-of-fold \(R^2\) of EEG-augmented models relative to the clinical baseline.

## Repository structure

```text
rsEEG_TDEHMM/
├── README.md
├── notebooks/
│   └── featureExtraction_acutestroke_fullGC_v2.ipynb
└── scripts/
    ├── feature_extraction_acutestroke.py
    ├── feature_extraction_acutestroke_v2.py
    ├── osl_dynemo_tms_eeg.py
    ├── osl_hmm_v005_processdata.py
    ├── osl_hmm_v006_trainingonly.py
    ├── osl_hmm_v006_trainingonly_v2.py
    ├── osl_hmm_v007_extract_features.py
    └── ted_hmm_functions_v002.py
```

The notebook contains the complete exploratory and post-hoc analysis workflow.
The scripts contain preprocessing, model-training and feature-extraction
utilities used to generate its inputs.

## Software environment

The code is written for Python 3 and uses packages including:

- `osl-dynamics`
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `nilearn`
- `openpyxl`

Exact compatibility depends on the OSL-Dynamics version used to train and save
the model. For reproducible execution, create an isolated environment and pin
the versions used in the original analysis before rerunning model training.

## Running the analysis

The scripts and notebook currently contain project-specific filesystem paths.
Before running them on another system:

1. replace the data, model and output paths with local paths;
2. install the required Python packages;
3. ensure that the source-space parcel ordering matches the DKT atlas mapping;
4. run preprocessing and HMM training before feature extraction;
5. execute the notebook in order, because later cells depend on objects created
   by earlier cells.

Large intermediate arrays, trained models and participant-level clinical files
are intentionally not included in this repository.

## Data availability and privacy

The underlying EEG and clinical data are not distributed here. Access is
subject to the approvals and data-governance requirements of the originating
study. Do not commit raw or derived participant-identifiable data to this
repository.

## Interpretation

The analyses are observational. Group differences and EEG–motor associations
should not be interpreted as causal. Predictive performance should be judged
from held-out, out-of-fold predictions and ultimately validated in an
independent cohort.

