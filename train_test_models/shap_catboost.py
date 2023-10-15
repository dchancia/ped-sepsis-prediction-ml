import numpy as np
import matplotlib.pyplot as plt
import shap
import os
from utils.preprocess_utils import preprocess_train_test, preprocess_train_val
from utils.shap_utils import shap_beeswarm, shap_scatter
from utils.threshold_utils import find_thresh
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")

# Selected features
col_names = ['albumin_mean', 'albumin_median', 'albumin_min', 'albumin_max', 'albumin_std', 
              'base_excess_mean', 'base_excess_median', 'base_excess_min', 'base_excess_max', 'base_excess_std', 
              'base_deficit_mean', 'base_deficit_median', 'base_deficit_min', 'base_deficit_max', 'base_deficit_std', 
              'art_pco2_mean', 'art_pco2_median', 'art_pco2_min', 'art_pco2_max', 'art_pco2_std', 
              'art_po2_mean', 'art_po2_median', 'art_po2_min', 'art_po2_max', 'art_po2_std', 
              'bicarbonate_mean', 'bicarbonate_median', 'bicarbonate_min', 'bicarbonate_max', 'bicarbonate_std', 
              'bilirubin_total_mean', 'bilirubin_total_median', 'bilirubin_total_min', 'bilirubin_total_max', 'bilirubin_total_std', 
              'bp_dias_mean', 'bp_dias_median', 'bp_dias_min', 'bp_dias_max', 'bp_dias_std', 
              'bp_sys_mean', 'bp_sys_median', 'bp_sys_min', 'bp_sys_max', 'bp_sys_std', 
              'bun_mean', 'bun_median', 'bun_min', 'bun_max', 'bun_std',
              'calcium_mean', 'calcium_median', 'calcium_min', 'calcium_max', 'calcium_std',
              'calcium_ionized_mean', 'calcium_ionized_median', 'calcium_ionized_min', 'calcium_ionized_max', 'calcium_ionized_std', 
              'chloride_mean', 'chloride_median', 'chloride_min', 'chloride_max', 'chloride_std', 
              'co2_mean', 'co2_median', 'co2_min', 'co2_max', 'co2_std', 
              'creatinine_mean', 'creatinine_median', 'creatinine_min', 'creatinine_max', 'creatinine_std', 
              'fio2_mean', 'fio2_median', 'fio2_min', 'fio2_max', 'fio2_std',
              'glucose_mean', 'glucose_median', 'glucose_min', 'glucose_max', 'glucose_std',
              'hemoglobin_mean', 'hemoglobin_median', 'hemoglobin_min', 'hemoglobin_max', 'hemoglobin_std', 
              'lactic_acid_mean', 'lactic_acid_median', 'lactic_acid_min', 'lactic_acid_max', 'lactic_acid_std', 
              'map_mean', 'map_median', 'map_min', 'map_max', 'map_std', 
              'pao2_fio2_mean', 'pao2_fio2_median', 'pao2_fio2_min', 'pao2_fio2_max', 'pao2_fio2_std', 
              'ph_mean', 'ph_median', 'ph_min', 'ph_max', 'ph_std', 
              'platelets_mean', 'platelets_median', 'platelets_min', 'platelets_max', 'platelets_std', 
              'potassium_mean', 'potassium_median', 'potassium_min', 'potassium_max', 'potassium_std', 
              'ptt_mean', 'ptt_median', 'ptt_min', 'ptt_max', 'ptt_std',
              'pulse_mean', 'pulse_median', 'pulse_min', 'pulse_max', 'pulse_std',
              'pupil_left_size_mean', 'pupil_left_size_median', 'pupil_left_size_min', 'pupil_left_size_max', 'pupil_left_size_std', 
              'pupil_left_reaction_Non-reactive', 'pupil_left_reaction_Not assessed', 'pupil_left_reaction_Reactive', 'pupil_left_reaction_Unable to Assess',
              'resp_mean', 'resp_median', 'resp_min', 'resp_max', 'resp_std',
              'sodium_mean', 'sodium_median', 'sodium_min', 'sodium_max', 'sodium_std', 
              'spo2_mean', 'spo2_median', 'spo2_min', 'spo2_max', 'spo2_std', 
              'temp_mean', 'temp_median', 'temp_min', 'temp_max', 'temp_std', 
              'wbc_mean', 'wbc_median', 'wbc_min', 'wbc_max', 'wbc_std', 
              'abnormal_heart_rate', 'abnormal_resp_rate', 'abnormal_temp', 'abnormal_wbc', 'abnormal_neut_bands', 'abnormal_bp_sys', 
              'abnormal_base_deficit', 'abnormal_lactate', 'cons_spo2_below90', 'fio2_above50', 'low_platelets', 'abnormal_pt', 'abnormal_inr', 
              'elevated_creat', 'abnormal_alt', 'abnormal_ast', 'on_asthma_meds', 'on_seizure_meds', 'on_insulin', 'psofa',
              'age_group_inf_tod', 'age_group_neonate', 'age_group_preschooler', 'age_group_school',  'label']
col_names_fixed = ['Albumin (Mean)', 'Albumin (Median)', 'Albumin (Min)', 'Albumin (Max)', 'Albumin (Std)', 
              'Base Excess (Mean)', 'Base Excess (Median)', 'Base Excess (Min)', 'Base Excess (Max)', 'Base Excess (Std)', 
              'Base Deficit (Mean)', 'Base Deficit (Median)', 'Base Deficit (Min)', 'Base Deficit (Max)', 'Base Deficit (Std)', 
              'Arterial PaCO2 (Mean)', 'Arterial PaCO2 (Median)', 'Arterial PaCO2 (Min)', 'Arterial PaCO2 (Max)', 'Arterial PaCO2 (Std)', 
              'Arterial PaO2 (Mean)', 'Arterial PaO2 (Median)', 'Arterial PaO2 (Min)', 'Arterial PaO2 (Max)', 'Arterial PaO2 (Std)', 
              'Bicarbonate (Mean)', 'Bicarbonate (Median)', 'Bicarbonate (Min)', 'Bicarbonate (Max)', 'Bicarbonate (Std)', 
              'Bilirubin (Mean)', 'Bilirubin (Median)', 'Bilirubin (Min)', 'Bilirubin (Max)', 'Bilirubin (Std)', 
              'Diastolic Blood Pressure (Mean)', 'Diastolic Blood Pressure (Median)', 'Diastolic Blood Pressure (Min)', 'Diastolic Blood Pressure (Max)', 'Diastolic Blood Pressure (Std)', 
              'Systolic Blood Pressure (Mean)', 'Systolic Blood Pressure (Median)', 'Systolic Blood Pressure (Min)', 'Systolic Blood Pressure (Max)', 'Systolic Blood Pressure (Std)', 
              'BUN (Mean)', 'BUN (Median)', 'BUN (Min)', 'BUN (Max)', 'BUN (Std)',
              'Calcium (Mean)', 'Calcium (Median)', 'Calcium (Min)', 'Calcium (Max)', 'Calcium (Std)',
              'Ionized Calcium (Mean)', 'Ionized Calcium (Median)', 'Ionized Calcium (Min)', 'Ionized Calcium (Max)', 'Ionized Calcium (Std)', 
              'Chloride (Mean)', 'Chloride (Median)', 'Chloride (Min)', 'Chloride (Max)', 'Chloride (Std)', 
              'CO2 (Mean)', 'CO2 (Median)', 'CO2 (Min)', 'CO2 (Max)', 'CO2 (Std)', 
              'Creatinine (Mean)', 'Creatinine (Median)', 'Creatinine (Min)', 'Creatinine (Max)', 'Creatinine (Std)', 
              'FiO2 (Mean)', 'FiO2 (Median)', 'FiO2 (Min)', 'FiO2 (Max)', 'FiO2 (Std)',
              'Glucose (Mean)', 'Glucose (Median)', 'Glucose (Min)', 'Glucose (Max)', 'Glucose (Std)',
              'Hemoglobin (Mean)', 'Hemoglobin (Median)', 'Hemoglobin (Min)', 'Hemoglobin (Max)', 'Hemoglobin (Std)', 
              'Lactic Acid (Mean)', 'Lactic Acid (Median)', 'Lactic Acid (Min)', 'Lactic Acid (Max)', 'Lactic Acid (Std)', 
              'Mean Arterial Pressure (Mean)', 'Mean Arterial Pressure (Median)', 'Mean Arterial Pressure (Min)', 'Mean Arterial Pressure (Max)', 'Mean Arterial Pressure (Std)', 
              'PaO2/FiO2 Ratio (Mean)', 'PaO2/FiO2 Ratio (Median)', 'PaO2/FiO2 Ratio (Min)', 'PaO2/FiO2 Ratio (Max)', 'PaO2/FiO2 Ratio (Std)', 
              'pH (Mean)', 'pH (Median)', 'pH (Min)', 'pH (Max)', 'pH (Std)', 
              'Platelets (Mean)', 'Platelets (Median)', 'Platelets (Min)', 'Platelets (Max)', 'Platelets (Std)', 
              'Potassium (Mean)', 'Potassium (Median)', 'Potassium (Min)', 'Potassium (Max)', 'Potassium (Std)', 
              'PTT (Mean)', 'PTT (Median)', 'PTT (Min)', 'PTT (Max)', 'PTT (Std)',
              'Heart Rate (Mean)', 'Heart Rate (Median)', 'Heart Rate (Min)', 'Heart Rate (Max)', 'Heart Rate (Std)',
              'Pupil Left Size (Mean)', 'Pupil Left Size (Median)', 'Pupil Left Size (Min)', 'Pupil Left Size (Max)', 'Pupil Left Size (Std)', 
              'Pupil Left Reaction (Non-reactive)', 'Pupil Left Reaction (Not assessed)', 'Pupil Left Reaction (Reactive)', 'Pupil Left Reaction (Unable to Assess)',
              'Respiratory Rate (Mean)', 'Respiratory Rate (Median)', 'Respiratory Rate (Min)', 'Respiratory Rate (Max)', 'Respiratory Rate (Std)',
              'Sodium (Mean)', 'Sodium (Median)', 'Sodium (Min)', 'Sodium (Max)', 'Sodium (Std)', 
              'SpO2 (Mean)', 'SpO2 (Median)', 'SpO2 (Min)', 'SpO2 (Max)', 'SpO2 (Std)', 
              'Temperature (Mean)', 'Temperature (Median)', 'Temperature (Min)', 'Temperature (Max)', 'Temperature (Std)', 
              'WBC (Mean)', 'WBC (Median)', 'WBC (Min)', 'WBC (Max)', 'WBC (Std)', 
              'Abnormal Heart Rate', 'Abnormal Respiratory Rate', 'Abnormal Temperature', 'Abnormal WBC', 'Abnormal Neutrophil Bands', 'Abnormal Systolic Blood Pressure', 
              'Abnormal Base Deficit', 'Abnormal Lactate', 'Consecutive SpO2 below 90', 'FiO2 above 50', 'Low Platelets', 'Abnormal PT', 'Abnormal INR', 
              'Elevated Creatinine', 'Abnormal ALT', 'Abnormal AST', 'On Asthma Meds', 'On Seizure Meds', 'On Insulin', 'pSOFA',
              'Age Group (Infant-Toddler)', 'Age Group (Neonate)', 'Age Group (Preschooler)', 'Age Group (School)',  'Label']

if __name__ == "__main__":

    # Define screening method
    path = '/home/dchanci/research/pediatric_sepsis/models/results'
    screening_methods = ['inf_psofa', 'sirs_od', 'inf_sirs_od']
    save_folder = 'results'

    # Iterate through screening methods
    for screening_method in screening_methods:

         # Define features for Shap scatter plots
        scatter_features = {'inf_psofa': ['pSOFA', 'FiO2 (Max)', 'Temperature (Max)', 'Systolic Blood Pressure (Min)', 'SpO2 (Median)', 'Heart Rate (Max)']}

        # Load data and preprocess derivation data
        X, y, X_train, X_test, y_train, y_test, pre_counter, pos_weight, post_counter, cols, X_test_non_norm = preprocess_train_test (os.path.join('/opt/moredata/dchanci/pediatric_sepsis/data_models/approach_m1', 'dataset_agg_eg_' + screening_method + '.csv'), col_names, col_names_fixed)

        # Fit model train-test split
        model = CatBoostClassifier(iterations=500, depth=8, verbose=False)
        model.fit(X_train, y_train)
        probas = model.predict_proba(X_test)
        new_thresh = find_thresh(y_test, probas, None, None)
        model.set_probability_threshold(new_thresh)

        # Generate shap plots test/holdout
        explainer = shap.TreeExplainer(model)
        shap_beeswarm(explainer, 'test', X_test, path, screening_method, save_folder)
        if screening_method == 'inf_psofa':
            shap_scatter(scatter_features, explainer, 'test', X_test, X_test_non_norm, probas[:,1], path, screening_method, save_folder)

        # Load data and preprocess validation data
        X_train, X_val, y_train, y_val, pos_weight, pre_counter, post_counter, X_train_non_norm, X_val_non_norm = preprocess_train_val (X, y, cols, os.path.join('/opt/moredata/dchanci/pediatric_sepsis/data_models/approach_m1', 'dataset_agg_sr_' + screening_method + '.csv'), col_names, col_names_fixed)

        # Fit model train dataset
        model = CatBoostClassifier(iterations=500, depth=8, verbose=False)
        model.fit(X_train, y_train)
        probas = model.predict_proba(X_val)
        new_thresh = find_thresh(y_val, probas, None, None)
        model.set_probability_threshold(new_thresh)

        # Generate shap plots train
        explainer = shap.TreeExplainer(model)
        shap_beeswarm(explainer, 'train', X_train, path, screening_method, save_folder)
        if screening_method == 'inf_psofa':
            shap_scatter(scatter_features, explainer, 'train', X_train, X_train_non_norm, y_train, path, screening_method, save_folder)

        # Generate shap plots val
        shap_beeswarm(explainer, 'val', X_val, path, screening_method, save_folder)
        if screening_method == 'inf_psofa':
            shap_scatter(scatter_features, explainer, 'val', X_val, X_val_non_norm, probas[:,1], path, screening_method, save_folder)
