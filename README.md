# ❤️ Heart Disease Risk Prediction

End-to-end machine learning project on the **UCI Heart Disease** dataset (303 patients, 13 clinical features): data cleaning, PCA, feature selection, supervised and unsupervised models, hyperparameter tuning, and a **Streamlit app** for interactive predictions.

`Python` · `Pandas` · `scikit-learn` · `Matplotlib` · `Streamlit`

## Results

Test set of 61 patients. Random Forest was the best baseline model and was saved as `final_model.pkl`.

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| **Random Forest** | **0.902** | 0.844 | **0.964** | **0.900** | **0.958** |
| Logistic Regression | 0.869 | 0.813 | 0.929 | 0.867 | 0.951 |
| SVM | 0.852 | 0.806 | 0.893 | 0.847 | 0.944 |
| Decision Tree | 0.721 | 0.657 | 0.821 | 0.730 | 0.729 |

**After tuning:** Random Forest (RandomizedSearchCV) reached **ROC AUC 0.962** and SVM (GridSearchCV) **0.960**, up from a 0.951 baseline.

High recall matters most here: the model should miss as few at-risk patients as possible.

## Pipeline

| Notebook | What it does |
|---|---|
| [`01_data_preprocessing`](01_data_preprocessing.ipynb) | Load the dataset, handle missing values, encode categories, create a binary target, EDA |
| [`02_pca_analysis`](02_pca_analysis.ipynb) | PCA, explained variance and a 2D projection |
| [`03_feature_selection`](03_feature_selection.ipynb) | Random Forest importance, RFE and Chi‑square selection |
| [`04_supervised_learning`](04_supervised_learning.ipynb) | Train and compare Logistic Regression, Decision Tree, Random Forest and SVM |
| [`05_unsupervised_learning`](05_unsupervised_learning.ipynb) | K‑Means (elbow method) and hierarchical clustering compared with the labels |
| [`06_hyperparameter_tuning`](06_hyperparameter_tuning.ipynb) | GridSearchCV / RandomizedSearchCV against the baseline |

Features selected by both RFE and Chi‑square include chest pain type (`cp`), exercise‑induced angina (`exang`), ST depression (`oldpeak`), slope, number of major vessels (`ca`) and `thal`.

## Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app loads `final_model.pkl`, takes the 13 patient attributes as input and returns the predicted probability of heart disease. To share it publicly from a notebook environment, expose port 8501 with ngrok (see `ngrok_setup.txt`).

> This project is for learning purposes only and is not a medical diagnostic tool.

## Files

```
01–06_*.ipynb        analysis notebooks, run in order
app.py               Streamlit prediction app
final_model.pkl      trained Random Forest pipeline
heart_disease.data   UCI Heart Disease (Cleveland) data
requirements.txt     Python dependencies
```

---

By **[Youssef Saeed](https://youssefsaeed-portfolio.vercel.app/)** · [LinkedIn](https://www.linkedin.com/in/youssef-saeed1/)
