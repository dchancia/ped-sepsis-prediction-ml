# Pediatric Sepsis Prediction

This repository provides code and models for a sepsis prediction model for a retrospective study for Pediatric Intensive Care Unit (PICU) patients admitted to Children's Healthcare of Atlanta (CHOA) from 2010 to 2022.

## Overview

![Graphical Abstract](./files/graphical_abstract.png)

## Sepsis Cohorts

To identify the required sepsis cohort, we followed 3 different screening approaches: **pSepsis-3**, SIRS + OD, and INF + SIRS + OD. The same inclusion criteria applied to all of them: children younger than 18 years old admitted at least once to the PICU during their hospitalization.

- pSepsis-3: Patients with suspected infection and pSOFA score $\ge$ 2 from 48 hours before to 24 hours after the infection time. Selected for final model.
- SIRS + OD: Patients with systemic inflammatory response syndrome and acute organ dysfrunction within a period of 24 hours.
- INF + SIRS + OD: Patients with suspected infection and SIRS + OD from 48 hours before to 24 hours after the infection time.

![Cohorts Flow Diagram](./files/flow_diagram.png)

The scripts for the screening approaches are in the folder [`screening_methods`](./screening_methods/) using the data generated with the scripts in the folder [`data_screening`](./data_screening/).

## Requirements

- Python >= 3.9.12
- `requirements.txt`

## Sepsis Prediction Model

We trained CatBoost, XGBoost, Random Forest, and Logistic Regression models for each of the 3 cohorts. The selected model, with the best performance, is CatBoost for the pSepsis-3 cohort. It outputs the risk of sepsis within the first seven days of hospital admission for PICU patients. Download the trained model: [`Sepsis Prediction model`](./models/sepsis_catboost.cbm).

### Features

<div align="center">

| Features |         |
|----------|---------|
| Vital signs     | Diastolic blood pressure<br>Systolic blood pressure<br>Mean arterial blood pressure<br>HeartRate<br>Respiratory rate <br>Oxygen saturation (SpO2)<br>Temperature |
| Laboratory Tests| Albumin<br>Base excess<br>Base deficit<br>Arterial PaCO2<br>Arterial PaO2<br>Bicarbonate<br>Bilirubin<br>Blood urea nitrogen (BUN)<br>Calcium<br>Ionized calcium<br>Chloride<br>Carbon dioxide (CO2)<br>Creatinine<br>Glucose<br>Hemoglobin<br>Lactic acid<br>pH<br>Platelets<br>Potassium<br>Partial thromboplastin time (PTT)<br>Sodium<br>White blood cell (WBC) count |
| Demographics                   | Age group                                     |
| Scoring Systems                | pSOFA                                         |
| Other Clinical Characteristics | Fraction inspired oxygen (FiO2)<br>PaO2/FiO2<br>Left pupil size<br>Left pupil reaction<br>Abnormal heart rate<br>Abnormal respiratory rate<br>Abnormal temperature<br>Abnormal WBC<br>Abnormal band neutrophils<br>Abnormal systolic blood pressure<br>Abnormal base deficit<br>Abnormal lactic acid<br>Abnormal prothrombin time (PT)<br>Abnormal international normalized ratio (INR)<br>Abnormal alanine aminotransferase (ALT)<br>Abnormal aspartate aminotransferase (AST)<br>Low platelets<br>Elevated creatinine<br>Two consecutive SpO2 <= 90<br>FiO2 > 50<br>On asthma medications<br>On seizure medications<br>On insulin |

</div>

### Preprocessing

The `get_features.ipynb` notebook in the `/data_models` folder shows the adopted preprocessing pipeline. It can be summarized in the following steps:

1. Collect data obtained within the first 7 days of hospital stay.
2. Resample data into two hour bins, using the median if there are multiple values recorded.
3. Discard outliers.
4. Impute missing values using forward fill for a patient-wise imputation, followed by population median imputation for the remaining missing values.
5. Select the 24-hour windows.
6. Aggregate data into a 24-hour bin using the mean, median, standard deviation, minimum, and maximum.
7. Add demographic information - age.
8. Add medications flags.

The `/train_test_models/utils/preprocess_utils.py` script contains multiple functions to split, normalize, and balance the dataset.

### Training

`/train_test_models/train_test_catboost.py`

### Validation

- Internal: Using the derivation dataset. 95% CI calculated using 50 different seeds for the train/test split. `/train_test_models/internal_val.py`.
- External: Using the external validation dataset. 95% CI calculated using bootstrapping. `/train_test_models/internal_val.py`.

### Interpretability

We generated shap beeswarm and scatter plots for the most important features using the the SHAP library. `/train_test_models/shap_catboost.py`.

<p float="left">
  <img src="./files/shap_external.png" width="42%" />
  <img src="./files/shap_external_scatter.png" width="53%" /> 
</p>