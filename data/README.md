# Data

`heart_disease.data` is the **Cleveland** file of the UCI Heart Disease dataset
(https://archive.ics.uci.edu/dataset/45/heart+disease): 303 patients, 13 clinical features and the diagnosis `num` (0 = no disease, 1–4 = disease).
Missing values are marked `?` (4 in `ca`, 2 in `thal`).

| Column | Meaning |
|---|---|
| age, sex | Age in years; sex (1 = male, 0 = female) |
| cp | Chest pain type: 1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic |
| trestbps, chol | Resting blood pressure (mm Hg); serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (1 = yes) |
| restecg | Resting ECG: 0 normal, 1 ST-T abnormality, 2 left ventricular hypertrophy |
| thalach, exang | Maximum heart rate; exercise-induced angina (1 = yes) |
| oldpeak, slope | ST depression induced by exercise; slope of the peak exercise ST segment (1 up, 2 flat, 3 down) |
| ca | Number of major vessels (0–3) coloured by fluoroscopy |
| thal | Thallium stress test: 3 normal, 6 fixed defect, 7 reversible defect |
| num | Diagnosis (target): 0 no disease, 1–4 disease |
