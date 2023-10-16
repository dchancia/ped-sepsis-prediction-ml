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

## Prediction Model
