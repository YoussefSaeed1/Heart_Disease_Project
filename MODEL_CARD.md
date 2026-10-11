# Model Card: Heart Disease Risk (logistic regression)

| | |
|---|---|
| **Model** | Logistic regression (C = 0.3) in a scikit-learn pipeline: median / most-frequent imputation, scaling, one-hot encoding |
| **Version** | 2.0.0 |
| **Owner** | Youssef Saeed |
| **Code** | `src/heart_disease/` · train with `python -m heart_disease.train` |
| **Metrics source** | [`reports/metrics.json`](reports/metrics.json), [`reports/data_validation.json`](reports/data_validation.json) |

## Intended use

**For:** learning and demonstrating a clean, validated classification workflow on a classic clinical dataset; showing how a single test split can mislead.

**Not for:** diagnosing, triaging or advising real patients. The data is from one hospital in the 1980s, the sample is small, and the model has not been clinically validated.

## Data

UCI Heart Disease, Cleveland file: 303 patients, 13 features, 46% with disease (diagnosis 1–4 grouped). 4 missing `ca` and 2 missing `thal` values are imputed inside the pipeline. Validation checks (ranges, known codes, duplicates, class balance) pass; see the README.

## Performance

| Evaluation | ROC AUC | Recall | Specificity |
|---|---:|---:|---:|
| 5-fold CV × 10 repeats (training set) | 0.899 ± 0.034 | – | – |
| Test set, 61 patients | 0.959 | 0.929 (26 / 28) | 0.848 (28 / 33) |
| Out-of-fold, all 303 patients (5-fold) | 0.915 | 0.791 | 0.902 |

The test set is easier than average; the cross-validated and out-of-fold figures are the realistic ones. The default 50% threshold misses about 1 in 5 patients with disease.

### By group (out-of-fold, 5-fold, threshold 50%)

| Group | Patients | With disease | ROC AUC | Recall | Specificity |
|---|---:|---:|---:|---:|---:|
| Men | 206 | 55% | 0.893 | 0.816 | 0.859 |
| Women | 97 | 26% | 0.929 | **0.680** | 0.958 |
| Under 55 | 143 | 31% | 0.929 | 0.773 | 0.939 |
| 55 and over | 160 | 59% | 0.885 | 0.800 | 0.846 |

**Women with disease are missed more often** (recall 68% vs 82% for men), mainly because they are fewer (97) and less often ill in this sample. Any real use would need a lower threshold or more data for women.

## Limitations and risks

- **Small, old, single-site sample.** 303 patients from Cleveland; results will not transfer to other populations without retraining.
- **Selection effect.** These patients were already referred for angiography, which is why "asymptomatic" chest pain has the highest disease rate.
- **Threshold choice.** 50% favours specificity over recall; for screening, a lower threshold (higher recall, more false alarms) is more appropriate.
- **Two features need a procedure.** `ca` (fluoroscopy) and `thal` (thallium test) are the strongest predictors and are only available after testing.

## Maintenance

`python -m heart_disease.train` re-runs validation, the model comparison, the figures and saves `models/heart_disease_model.joblib`. Tests: `pytest`.
