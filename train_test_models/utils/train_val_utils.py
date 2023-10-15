import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, average_precision_score, roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.calibration import calibration_curve
from utils.threshold_utils import find_thresh, get_coord
import warnings
warnings.filterwarnings("ignore")
import scipy.stats as st


def internal_val (classifiers, X, y, names_classifiers, fixed, colors, path, save_folder, screening_method, reps, col_names_fixed):    
    """
    Train and evaluate models multiple times with derivation data (Egleston)
    using different seeds for the train-test split.

    It generates the evaluation metrics, and ROC, PR, and Calibration curves.
    """

    cols = col_names_fixed[:-1]

    undersampler = RandomUnderSampler(sampling_strategy=0.3, random_state=42)
    scaler = StandardScaler()

    fig1 = plt.figure(figsize=(7,7))
    ax1 = fig1.add_subplot(1, 1, 1)

    fig2 = plt.figure(figsize=(7,7))
    ax2 = fig2.add_subplot(1, 1, 1)

    fig3 = plt.figure(figsize=(7,7))
    ax3 = fig3.add_subplot(1, 1, 1)

    metrics = []
        
    for j in range(len(classifiers)):

        clf = classifiers[j]
        name = names_classifiers[j]

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        auprcs = []
        precisions = []
        y_real = []
        y_proba = []
        mean_recall = np.linspace(0, 1, 100)

        prob_true_list = []
        prob_pred_list = []

        for i in range(reps):

            print(name, i + 1)

            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i, stratify=y)

            # Normalize
            X_train = pd.DataFrame(scaler.fit_transform(X_train))
            X_test = pd.DataFrame(scaler.transform(X_test))
            X_train.columns = cols
            X_test.columns = cols

            # Balance training set
            X_train, y_train = undersampler.fit_resample(X_train, y_train)
            
            # Fit model
            clf.fit(X_train, y_train)
            probas = clf.predict_proba(X_test)
            preds = clf.predict(X_test)

            if fixed == True:
                new_thresh = find_thresh(y_test, probas, i, reps)
                print(new_thresh)
                if new_thresh != None:
                    if name == 'CatBoost':
                        clf.set_probability_threshold(new_thresh)
                        preds = clf.predict(X_test)
                    else:
                        preds = [1 if x >= new_thresh else 0 for x in list(probas[:,1])]
                else:
                    continue

            # Compute evaluation metrics
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            recall = recall_score(y_test, preds)
            precision = precision_score(y_test, preds)
            tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
            specificity = tn / (tn + fp)
            ppv = tp / (tp + fp)
            npv = tn / (tn + fn)
            auc_score = roc_auc_score(y_test, probas[:,1])
            auprc_score = average_precision_score(y_test, probas[:,1])
            metrics.append([name, acc, f1, recall, precision, specificity, auc_score, auprc_score, npv, ppv])

            # Compute ROC curve and area under the curve
            fpr, tpr, _ = roc_curve(y_test, probas[:, 1])
            tprs.append(np.interp(mean_fpr, fpr, tpr))            
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(auc_score)
            # ax1.plot(fpr, tpr, lw=0.8, alpha=0.3, c=colors[name])

            # Compute PR curve and area under the curve
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, probas[:, 1])
            precision_curve_inv, recall_curve_inv = precision_curve[::-1], recall_curve[::-1]
            precisions.append(np.interp(mean_recall, recall_curve_inv, precision_curve_inv))
            auprcs.append(auprc_score)
            y_real.append(y_test)
            y_proba.append(probas[:, 1])
            # ax2.plot(recall_curve, precision_curve, lw=0.8, alpha=0.3, c=colors[name])

            # Compute calibration curve
            prob_true, prob_pred = calibration_curve(y_test, probas[:,1], n_bins=10)
            if i == 0:
                prob_true_len = len(prob_true)
            if len(prob_true) == prob_true_len:
                prob_pred_list.append(prob_pred)
                prob_true_list.append(prob_true)
            # ax3.plot(prob_pred, prob_true, lw=0.8, alpha=0.3, c=colors[name])

        # Plot AUROC
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        ci = st.norm.interval(alpha=0.95, loc=mean_auc, scale=st.sem(aucs))
        ci_l = round(ci[0], 2)
        ci_u = round(ci[1], 2)
        if str(ci_l) == 'nan':
            ci_l = mean_auc
        if str(ci_u) == 'nan':
            ci_u = mean_auc
        ax1.plot(mean_fpr, mean_tpr, c=colors[name], label=r'%s Mean AUROC: %0.2f (%0.2f, %0.2f)' % (name, mean_auc, ci_l, ci_u), lw=1.5, alpha=1)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax1.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[name], alpha=0.1)
        coord_x, coord_y = get_coord(mean_fpr, mean_tpr)
        ax1.scatter(coord_x, coord_y, c=colors[name], marker='+', linewidths=1, s=1000)

        # Plot AUPRC
        y_real = np.concatenate(y_real)
        y_proba = np.concatenate(y_proba)
        precision_curve, recall_curve, _ = precision_recall_curve(y_real, y_proba)
        baseline = len(y_test[y_test==1]) / len(y_test)
        mean_auprc = np.mean(auprcs)
        ci = st.norm.interval(alpha=0.95, loc=mean_auprc, scale=st.sem(auprcs))
        ci_l = round(ci[0], 2)
        ci_u = round(ci[1], 2)
        if str(ci_l) == 'nan':
            ci_l = mean_auprc
        if str(ci_u) == 'nan':
            ci_u = mean_auprc
        ax2.plot(recall_curve, precision_curve, c=colors[name], label=r'%s Mean AUPRC: %0.2f (%0.2f, %0.2f)' % (name, mean_auprc, ci_l, ci_u), lw=1.5, alpha=1) 
        mean_precision = np.mean(precisions, axis=0)
        std_precision = np.std(precisions, axis=0)
        ax2.fill_between(mean_recall, mean_precision + std_precision, mean_precision - std_precision, color=colors[name], alpha=0.1)

        # Plot calibration curve
        mean_prob_pred = np.mean(prob_pred_list, axis=0)
        mean_prob_true = np.mean(prob_true_list, axis=0)
        ax3.plot(mean_prob_pred, mean_prob_true, color=colors[name], lw=1.5, label=name, alpha=1)
        std_prob_true = np.std(prob_true_list, axis=0)
        prob_true_upper = np.minimum(mean_prob_true + std_prob_true, 1)
        prob_true_lower = np.maximum(mean_prob_true - std_prob_true, 0)
        ax3.fill_between(mean_prob_pred, prob_true_lower, prob_true_upper, color=colors[name], alpha=0.1)
        

    # Plot AUROC

    ax1.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray', label='No Skill', alpha=.8)
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.set_xlabel('False Positive Rate', fontsize=18)
    ax1.set_ylabel('True Positive Rate', fontsize=18)
    ax1.set_title('Receiver Operating Characteristic Curve', fontsize=18)
    ax1.legend(loc="lower right", fontsize=12)
    if fixed == True:
        fig1.savefig(os.path.join(path, screening_method, save_folder, 'roc_holdout_fixed.png'))
    else:
        fig1.savefig(os.path.join(path, screening_method, save_folder, 'roc_holdout.png'))

    # Plot AUPRC
    ax2.plot([0, 1], [baseline, baseline], linestyle='--', lw=1, color='gray', label='No Skill', alpha=.8)
    ax2.set_xlim([-0.05, 1.05])
    ax2.set_ylim([-0.05, 1.05])
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.set_xlabel('Recall', fontsize=18)
    ax2.set_ylabel('Precision', fontsize=18)
    ax2.set_title('Precision-Recall Curve', fontsize=18)
    ax2.legend(loc="lower left", fontsize=12)
    if fixed == True:
        fig2.savefig(os.path.join(path, screening_method, save_folder, 'pr_holdout_fixed.png'))
    else:
        fig2.savefig(os.path.join(path, screening_method, save_folder, 'pr_holdout.png'))

    # Plot calibration curve
    ax3.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray', label='Perfectly calibrated', alpha=.8)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.set_xlabel('Average Predicted Probability', fontsize=18)
    ax3.set_ylabel('Ratio of Positives', fontsize=18)
    ax3.set_title('Calibration Curve', fontsize=18)
    ax3.legend(loc="upper left", fontsize=12)
    if fixed == True:
        fig3.savefig(os.path.join(path, screening_method, save_folder, 'cal_holdout_fixed.png'))
    else:
        fig3.savefig(os.path.join(path, screening_method, save_folder, 'cal_holdout.png'))

    return metrics



def external_val (classifiers, X, y, data, names_classifiers, fixed, colors, path, save_folder, screening_method, reps, col_names_fixed):    
    """
    Train and evaluate models multiple times with derivation data (Egleston) 
    and external validation data (Scottish Rite) using the bootstrap method.

    It generates the evaluation metrics, and ROC, PR, and Calibration curves.
    """

    cols = col_names_fixed[:-1]

    undersampler = RandomUnderSampler(sampling_strategy=0.3, random_state=42)
    scaler = StandardScaler()

    fig1 = plt.figure(figsize=(7,7))
    ax1 = fig1.add_subplot(1, 1, 1)

    fig2 = plt.figure(figsize=(7,7))
    ax2 = fig2.add_subplot(1, 1, 1)

    fig3 = plt.figure(figsize=(7,7))
    ax3 = fig3.add_subplot(1, 1, 1)

    metrics_true = []
    metrics = []
        
    for j in range(len(classifiers)):

        clf = classifiers[j]
        name = names_classifiers[j]

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        auprcs = []
        precisions = []
        y_real = []
        y_proba = []
        mean_recall = np.linspace(0, 1, 100)

        prob_true_list = []
        prob_pred_list = []

        y_train = y

        # Normalize
        X_train = pd.DataFrame(scaler.fit_transform(X))
        X_train.columns = cols

        # Balance training set
        X_train, y_train = undersampler.fit_resample(X_train, y_train)
        
        # Fit model
        clf.fit(X_train, y_train)

        for i in range(reps):

            print(name, i + 1)

            # Create dataset
            # if i == reps-1:
            if i == 15:
                y_test = data['label']
                X_test = data.drop('label', axis=1)
            else:
                bootstr = data.sample(frac=1, replace=True)
                y_test = bootstr['label']
                X_test = bootstr.drop('label', axis=1)

            X_test.columns = cols

            # Normalize
            X_test = pd.DataFrame(scaler.transform(X_test))
            X_test.columns = cols
            
            # Validate model
            probas = clf.predict_proba(X_test)
            preds = clf.predict(X_test)

            if fixed == True:
                new_thresh = find_thresh(y_test, probas, i, reps)
                print(new_thresh)
                if new_thresh != None:
                    if name == 'CatBoost':
                        clf.set_probability_threshold(new_thresh)
                        preds = clf.predict(X_test)
                    else:
                        preds = [1 if x >= new_thresh else 0 for x in list(probas[:,1])]
                else:
                    continue

            # Compute evaluation metrics
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            recall = recall_score(y_test, preds)
            precision = precision_score(y_test, preds)
            tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
            specificity = tn / (tn + fp)
            ppv = tp / (tp + fp)
            npv = tn / (tn + fn)
            auc_score = roc_auc_score(y_test, probas[:,1])
            auprc_score = average_precision_score(y_test, probas[:,1])
            metrics.append([name, acc, f1, recall, precision, specificity, auc_score, auprc_score, npv, ppv])
            # if i == reps-1:
            if fixed == False:
                metrics_true.append([name, acc, f1, recall, precision, specificity, auc_score, auprc_score, npv, ppv])
            elif fixed == True and new_thresh != None:
                metrics_true.append([name, acc, f1, recall, precision, specificity, auc_score, auprc_score, npv, ppv])

            # Compute ROC curve and area under the curve
            fpr, tpr, _ = roc_curve(y_test, probas[:, 1])
            tprs.append(np.interp(mean_fpr, fpr, tpr))            
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(auc_score)

            # Compute PR curve and area under the curve
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, probas[:, 1])
            precision_curve_inv, recall_curve_inv = precision_curve[::-1], recall_curve[::-1]
            precisions.append(np.interp(mean_recall, recall_curve_inv, precision_curve_inv))
            auprcs.append(auprc_score)
            y_real.append(y_test)
            y_proba.append(probas[:, 1])

            # Compute calibration curve
            prob_true, prob_pred = calibration_curve(y_test, probas[:,1], n_bins=10)
            if i == 0:
                prob_true_len = len(prob_true)
            if len(prob_true) == prob_true_len:
            # if len(prob_true) == 10:
                prob_pred_list.append(prob_pred)
                prob_true_list.append(prob_true)
        
        # Plot AUROC
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        ci = st.norm.interval(alpha=0.95, loc=auc_score, scale=st.sem(aucs))
        ci_l = round(ci[0], 2)
        ci_u = round(ci[1], 2)
        if str(ci_l) == 'nan':
            ci_l = auc_score
        if str(ci_u) == 'nan':
            ci_u = auc_score
        ax1.plot(fpr, tpr, c=colors[name], label=r'%s Mean AUROC: %0.2f (%0.2f, %0.2f)' % (name, auc_score, ci_l, ci_u), lw=1.5, alpha=1)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax1.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[name], alpha=0.1)
        coord_x, coord_y = get_coord(fpr, tpr)
        ax1.scatter(coord_x, coord_y, c=colors[name], marker='+', linewidths=1, s=1000)

        # Plot AUPRC
        y_real = np.concatenate(y_real)
        y_proba = np.concatenate(y_proba)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, probas[:, 1])
        baseline = len(y_test[y_test==1]) / len(y_test)
        ci = st.norm.interval(alpha=0.95, loc=auprc_score, scale=st.sem(auprcs))
        ci_l = round(ci[0], 2)
        ci_u = round(ci[1], 2)
        if str(ci_l) == 'nan':
            ci_l = auprc_score
        if str(ci_u) == 'nan':
            ci_u = auprc_score
        ax2.plot(recall_curve, precision_curve, c=colors[name], label=r'%s Mean AUPRC: %0.2f (%0.2f, %0.2f)' % (name, auprc_score, ci_l, ci_u), lw=1.5, alpha=1) 
        mean_precision = np.mean(precisions, axis=0)
        std_precision = np.std(precisions, axis=0)
        ax2.fill_between(mean_recall, mean_precision + std_precision, mean_precision - std_precision, color=colors[name], alpha=0.1)

        # Plot calibration curve
        mean_prob_pred = np.mean(prob_pred_list, axis=0)
        mean_prob_true = np.mean(prob_true_list, axis=0)
        prob_true, prob_pred = calibration_curve(y_test, probas[:,1], n_bins=10)
        ax3.plot(prob_pred, prob_true, color=colors[name], lw=1.5, label=name, alpha=1)
        std_prob_true = np.std(prob_true_list, axis=0)
        prob_true_upper = np.minimum(mean_prob_true + std_prob_true, 1)
        prob_true_lower = np.maximum(mean_prob_true - std_prob_true, 0)
        ax3.fill_between(mean_prob_pred, prob_true_lower, prob_true_upper, color=colors[name], alpha=0.1)
   

    # Plot AUROC
    ax1.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray', label='No Skill', alpha=.8)
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.set_xlabel('False Positive Rate', fontsize=18)
    ax1.set_ylabel('True Positive Rate', fontsize=18)
    ax1.set_title('Receiver Operating Characteristic Curve', fontsize=18)
    ax1.legend(loc="lower right", fontsize=12)
    if fixed == True:
        fig1.savefig(os.path.join(path, screening_method, save_folder, 'roc_external_fixed.png'))
    else:
        fig1.savefig(os.path.join(path, screening_method, save_folder, 'roc_external.png'))

    # Plot AUPRC
    baseline = len(y_test[y_test==1]) / len(y_test)
    ax2.plot([0, 1], [baseline, baseline], linestyle='--', lw=1, color='gray', label='No Skill', alpha=.8)
    ax2.set_xlim([-0.05, 1.05])
    ax2.set_ylim([-0.05, 1.05])
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.set_xlabel('Recall', fontsize=18)
    ax2.set_ylabel('Precision', fontsize=18)
    ax2.set_title('Precision-Recall Curve', fontsize=18)
    ax2.legend(loc="lower left", fontsize=12)
    if fixed == True:
        fig2.savefig(os.path.join(path, screening_method, save_folder, 'pr_external_fixed.png'))
    else:
        fig2.savefig(os.path.join(path, screening_method, save_folder, 'pr_external.png'))

    # Plot calibration curve
    ax3.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray', label='Perfectly calibrated', alpha=.8)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.set_xlabel('Average Predicted Probability', fontsize=18)
    ax3.set_ylabel('Ratio of Positives', fontsize=18)
    ax3.set_title('Calibration Curve', fontsize=18)
    ax3.legend(loc="upper left", fontsize=12)
    if fixed == True:
        fig3.savefig(os.path.join(path, screening_method, save_folder, 'cal_external_fixed.png'))
    else:
        fig3.savefig(os.path.join(path, screening_method, save_folder, 'cal_external.png'))

    return metrics, metrics_true

