# ICU-Ventilation-Duration-Prediction

# 🧠 Data-Driven Prediction of Ventilation Duration in ICU Patients

### DS5003 Healthcare Data Science — Team 2

**University of Virginia | School of Data Science**

**Team Members:**

* Harold Haugen – [waa4bq@virginia.edu](mailto:waa4bq@virginia.edu)
* Alec Pixton – [etk3pu@virginia.edu](mailto:etk3pu@virginia.edu)
* Clarissa Benitez – [ycv3fh@virginia.edu](mailto:ycv3fh@virginia.edu)
* Emmanuel Leonce – [fyb7sx@virginia.edu](mailto:fyb7sx@virginia.edu)

---

## 📋 Overview

This project explores the use of **machine learning** to predict the **duration of mechanical ventilation** required by ICU patients. By accurately forecasting which patients will need prolonged respiratory support, hospitals can better allocate ventilators, staff, and ICU resources — improving patient outcomes and operational efficiency.

Our work leverages the **MIMIC-III Clinical Database**, containing de-identified data from ICU stays at Beth Israel Deaconess Medical Center (2001–2012). The dataset includes vital signs, lab results, and demographic information collected within the first 24 hours of ICU admission.

---

## 🎯 Objective

Develop a predictive model that classifies whether an ICU patient will require **mechanical ventilation for more than one day**.

Key goals include:

* Supporting ICU resource planning and triage.
* Identifying high-risk patients early.
* Evaluating and comparing statistical and machine learning methods.

---

## 👥 Target Audience

* Chief Operating Officer (COO)
* Director of Operations & Patient Flow Coordinators
* ICU Nursing Supervisors & Clinicians (Pulmonologists, Respiratory Therapists)
* Hospital Resource and Capacity Managers

---

## 🧩 Data Sources

**Primary Source:** [MIMIC-III Clinical Database v1.4](https://physionet.org/content/mimiciii/1.4/)

**Key Tables Used:**

| Table                | Description                   | Rows    | Columns |
| -------------------- | ----------------------------- | ------- | ------- |
| `PROCEDUREEVENTS_MV` | Mechanical ventilation events | 258,066 | 25      |
| `ADMISSIONS`         | Admission metadata            | 58,976  | 19      |
| `PATIENTS`           | Demographics                  | 46,520  | 8       |
| `LABEVENTS`          | Laboratory results            | 27.8M   | 9       |
| `CHARTEVENTS`        | Vital signs and charted data  | 330.7M  | 15      |
| `ICUSTAYS`           | ICU stay records              | 61,532  | 12      |

A total of **8,392 unique ventilation events** and **73 features** were used after preprocessing and merging.

---

## ⚙️ Methodology

### 1. Data Preparation

* Identified intubation and ventilation records.
* Aggregated lab, chart, and demographic data by `SUBJECT_ID` and `HADM_ID`.
* Created engineered features such as:

  * `Resp_Diag_Label` (respiratory diagnosis flag)
  * `Age_Admission` (adjusted for de-identification)
  * Scaled and one-hot encoded categorical variables.

### 2. Modeling Approach

A **binary classification task** was defined:

* **0** = Ventilation ≤ 1 day
* **1** = Ventilation > 1 day

**Models Tested:**

* Logistic Regression
* Random Forest
* XGBoost
* Neural Network

Performance was evaluated on both **full** and **invasive-only** subsets using an 80/20 train-test split.

### 3. Evaluation Metrics

* **Primary metric:** Recall (positive class = >1 day ventilation)
* Secondary metrics: ROC-AUC, Precision, Specificity, Accuracy

---

## 📊 Results

| Model               | Dataset | ROC-AUC   | Recall    | Specificity | Precision | Accuracy |
| ------------------- | ------- | --------- | --------- | ----------- | --------- | -------- |
| Random Forest       | Full    | **0.805** | **0.819** | 0.652       | 0.697     | 0.735    |
| XGBoost             | Full    | 0.823     | 0.811     | 0.692       | 0.726     | 0.752    |
| Logistic Regression | Full    | 0.792     | 0.727     | 0.729       | 0.726     | 0.724    |
| Neural Network      | Full    | 0.796     | 0.754     | 0.703       | 0.713     | 0.728    |

> ✅ **Best model:** Random Forest (Recall = 0.819)

---

## 🧠 Feature Importance

Top predictive features (from SHAP analysis):

* **PO2 (Partial Pressure of Oxygen)**
* **Mean Airway Pressure (MAP)**
* **Heart Rate**
* **Age at Admission**
* **Respiratory Diagnosis Flag**

---

## 📈 Key Insights

* Machine learning models can **reliably classify** ventilation duration based on early ICU indicators.
* Ensemble methods (Random Forest, XGBoost) outperform simpler models in recall and ROC-AUC.
* Feature importance analysis revealed that **vital signs and respiratory biomarkers** are critical predictors.

---

## ⚠️ Limitations

* Limited sample subset of MIMIC-III (8,392 records).
* Retrospective analysis limits generalizability.
* Lack of time-series and real-world intervention data.

---

## 🔮 Future Work

* Expand to full MIMIC-III and MIMIC-IV datasets.
* Integrate longitudinal and temporal features.
* Collaborate with clinicians to enhance feature engineering.
* Explore threshold optimization to increase recall in deployment.

---

## 🧾 References

1. [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/)
2. [Mechanical Ventilation in the ICU – AAST](https://www.aast.org/resources-detail/mechanical-ventilation-in-intensive-care-unit)
3. [Nature (2025): COVID-19 Ventilator Burden Study](https://www.nature.com/articles/s41598-025-99863-3)
4. [XGBoost Feature Importance Guide](https://medium.com/@emilykmarsh/xgboost-feature-importance-233ee27c33a4)
5. [SHAP Documentation](https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Basic%20SHAP%20Interaction%20Value%20Example%20in%20XGBoost.html)

---

## 💻 Repository Structure

```
📦 Ventilation-Duration-Prediction
├── data/
│   ├── raw/                # Unprocessed MIMIC-III extracts
│   ├── processed/          # Cleaned and feature-engineered data
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_feature_importance.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── evaluate.py
├── results/
│   ├── figures/
│   ├── metrics/
├── README.md
└── requirements.txt
```

---

## 🧬 Tech Stack

* **Python 3.10+**
* `pandas`, `numpy`, `scikit-learn`, `xgboost`, `tensorflow`, `shap`, `matplotlib`

---

## 📜 License

This repository is for **academic and educational use** under the MIT License.

---
