import matplotlib.pyplot as plt
import numpy as np
import shap
import os
import warnings
warnings.filterwarnings("ignore")


def shap_beeswarm(explainer, split, data, path, screening_method, save_folder):
    """
    Generate beeswarm plot with SHAP library.
    """
    
    shap_values = explainer.shap_values(data)
    plt.figure()
    shap.summary_plot(shap_values, data, show=False, plot_size=(8,10), max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(path, screening_method, save_folder, 'shap_' + split + '.png'))


def shap_scatter(scatter_features, explainer, split, data, data_non_norm, color, path, screening_method, save_folder):
    """
    Generate scatter plots with SHAP library.
    """

    shap_values = explainer(data)
    shap_values.data = np.array(data_non_norm)
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10,10))
    n = 0
    m = 0
    for i in range(len(scatter_features[screening_method])):
        feature = scatter_features[screening_method][i]
        shap.plots.scatter(shap_values[:, feature], color=np.array(color), ax=axes[n, m], show=False)
        n += 1
        if i == 2:
            n = 0
            m += 1
    plt.tight_layout()
    plt.savefig(os.path.join(path, screening_method, save_folder, 'shap_' + split + '_scatter.png'))

