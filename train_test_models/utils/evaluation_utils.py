from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, roc_curve
import pandas as pd
import numpy as np
import scipy.stats as st
import warnings
warnings.filterwarnings("ignore")


def eval_metrics(y_test, preds, probs):
    """
    Compute widely used evaluation metrics.
    """
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    recall = recall_score(y_test, preds)
    precision = precision_score(y_test, preds)
    auc_score = roc_auc_score(y_test, probs[:,1])
    cm = confusion_matrix(y_test, preds, normalize=None)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn)
    ppv = tp / (tp + fp)
    fpr, tpr, thresh = roc_curve(y_test, probs[:,1], pos_label=1)

    print('Accuracy:', round(acc, 2))
    print('F1-Score:', round(f1, 2))
    print('Recall:', round(recall, 2))
    print('Precision:', round(precision, 2))
    print('AUC Score:', round(auc_score, 2))
    print('Specificity:', round(specificity, 2))
    print('NPV:', round(npv, 2))
    print('PPV:', round(ppv, 2))
    print('Confusion Matrix:')
    print(cm)



def compute_metrics (metrics_df, metrics_true_df, names):
    """
    Generate dataframe with all evaluation metrics with the corresponding
    95% confidence interval for multiple models.
    """
    
    metrics = pd.DataFrame(columns=metrics_df.columns)
    for i in range(len(names)):
        data = metrics_df[metrics_df['Model'] == names[i]]
        if isinstance(metrics_true_df, pd.DataFrame):
            data_true = metrics_true_df[metrics_true_df['Model'] == names[i]]
        else:
            data_true = None
        metrics.loc[i, 'Model'] = names[i]
        for col in data.columns[1:]:
            if isinstance(data_true, pd.DataFrame):
                mean_metric = round(data_true[col].values[0], 2)
                ci = st.norm.interval(confidence=0.95, loc=data_true[col].values[0], scale=st.sem(data[col]))                
            else:
                mean_metric = round(np.mean(data[col]), 2)
                ci = st.norm.interval(confidence=0.95, loc=np.mean(data[col]), scale=st.sem(data[col]))
            ci_b = round(ci[0], 2)
            ci_u = round(ci[1], 2)
            if str(ci_b) == 'nan':
                ci_b = mean_metric
            if str(ci_u) == 'nan':
                ci_u = mean_metric
            metrics.loc[i, col] = str(mean_metric) + ' (' + str(ci_b) + ', ' + str(ci_u) + ')'

    return metrics