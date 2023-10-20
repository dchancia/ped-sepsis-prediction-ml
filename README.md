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

We trained CatBoost, XGBoost, Random Forest, and Logistic Regression models for each of the 3 cohorts. The selected model, with the best performance, is CatBoost for the pSepsis-3 cohort. It outputs the risk of sepsis within the first seven days of hospital admission for PICU patients. Download the trained model: [`sepsis_catboost.cbm`](./models/sepsis_catboost.cbm)

### Features

| Features |         |
|----------|---------|
| Vital signs     | Diastolic blood pressure<br>Systolic blood pressure<br>Mean arterial blood pressure<br>HeartRate<br>Respiratory rate <br>Oxygen saturation (SpO2)<br>Temperature |
| Laboratory Tests| Albumin<br>Base excess<br>Base deficit<br>Arterial PaCO2<br>Arterial PaO2<br>Bicarbonate<br>Bilirubin<br>Blood urea nitrogen (BUN)<br>Calcium<br>Ionized calcium<br>Chloride<br>Carbon dioxide (CO2)<br>Creatinine<br>Glucose<br>Hemoglobin<br>Lactic acid<br>pH<br>Platelets<br>Potassium<br>Partial thromboplastin time (PTT)<br>Sodium<br>White blood cell (WBC) count |
| Demographics                   | Age group                                     |
| Scoring Systems                | pSOFA                                         |
| Other Clinical Characteristics | Fraction inspired oxygen (FiO2)<br>PaO2/FiO2<br>Left pupil size<br>Left pupil reaction<br>Abnormal heart rate<br>Abnormal respiratory rate<br>Abnormal temperature<br>Abnormal WBC<br>Abnormal band neutrophils<br>Abnormal systolic blood pressure<br>Abnormal base deficit<br>Abnormal lactic acid<br>Abnormal prothrombin time (PT)<br>Abnormal international normalized ratio (INR)<br>Abnormal alanine aminotransferase (ALT)<br>Abnormal aspartate aminotransferase (AST)<br>Low platelets<br>Elevated creatinine<br>Two consecutive SpO2 <= 90<br>FiO2 > 50<br>On asthma medications<br>On seizure medications<br>On insulin |