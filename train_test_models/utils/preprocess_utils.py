import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


def preprocess_der (path, col_names, col_names_fixed):
    """
    Preprocess the derivation dataset (Egleston) for internal validation.
    """

    # Load data
    data = pd.read_parquet(path)
    data = data.replace(np.nan, 0)
    data = data.replace([np.inf, -np.inf], 0).dropna()

    # Add age groups
    data['age_group'] = np.where(data['age_years'] <= 0.083, 'neonate', np.where(data['age_years'] <= 3.0, 'inf_tod', 
                        np.where(data['age_years'] <= 6.0, 'preschooler', 'school')))
    data.drop('age_years', axis=1, inplace=True)

    # Create dummies
    data = pd.get_dummies(data, columns=['pupil_right_reaction', 'pupil_left_reaction', 'age_group'])
    data = data.drop(['patid', 'csn'], axis=1)

    # Select columns
    data = data[col_names]

    # Create dataset
    y = data['label']
    X = data.drop('label', axis=1)
    X.columns = col_names_fixed[:-1]

    return X, y



def preprocess_val (path, col_names):
    """
    Preprocess the external validation dataset (Scottish Rite) for external validation.
    """

    # Load data
    data = pd.read_parquet(path)
    data = data.replace(np.nan, 0)
    data = data.replace([np.inf, -np.inf], 0).dropna()

    # Add age groups
    data['age_group'] = np.where(data['age_years'] <= 0.083, 'neonate', np.where(data['age_years'] <= 3.0, 'inf_tod', 
                        np.where(data['age_years'] <= 6.0, 'preschooler', 'school')))
    data.drop('age_years', axis=1, inplace=True)

    # Create dummies
    data = pd.get_dummies(data, columns=['pupil_right_reaction', 'pupil_left_reaction', 'age_group'])
    data = data.drop(['patid', 'csn'], axis=1)

    # Select columns
    data_val = data[col_names]

    return data_val



def preprocess_train_test (path, col_names, col_names_fixed):
    """
    Preprocess train-test splits from derivation dataset (Egleston).
    """

    data = pd.read_parquet(path)
    data = data.replace(np.nan, 0)
    data = data.replace([np.inf, -np.inf], 0).dropna()

    # Add age groups
    data['age_group'] = np.where(data['age_years'] <= 0.083, 'neonate', np.where(data['age_years'] <= 3.0, 'inf_tod', 
                        np.where(data['age_years'] <= 6.0, 'preschooler', 'school')))
    data.drop('age_years', axis=1, inplace=True)

    # Create dummies
    data = pd.get_dummies(data, columns=['pupil_right_reaction', 'pupil_left_reaction', 'age_group'])
    data = data.drop(['patid', 'csn'], axis=1)

    # Select columns
    data = data[col_names]

    # Create dataset
    y = data['label']
    X = data.drop('label', axis=1)
    X.columns = col_names_fixed[:-1]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_test_non_norm = X_test.copy()
    cols = X_train.columns

    # Normalize
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))
    X_train.columns = cols
    X_test.columns = cols

    # Summarize class distribution training split
    pre_counter = Counter(y_train)
    pos_weight = pre_counter[0] / pre_counter[1]

    # Balance training split (undersampling)
    undersampler = RandomUnderSampler(sampling_strategy=0.3, random_state=42)
    X_train, y_train = undersampler.fit_resample(X_train, y_train)
    post_counter = Counter(y_train)

    return X, y, X_train, X_test, y_train, y_test, pre_counter, pos_weight, post_counter, cols, X_test_non_norm



def preprocess_train_val (X, y, cols, path, col_names, col_names_fixed):
    """
    Preprocess train and validation datasets from derivation dataset (Egleston)
    and external validation dataset (Scottish Rite).
    """

    # Load data
    data = pd.read_parquet(path)
    data = data.replace(np.nan, 0)
    data = data.replace([np.inf, -np.inf], 0).dropna()

    # Add age groups
    data['age_group'] = np.where(data['age_years'] <= 0.083, 'neonate', np.where(data['age_years'] <= 3.0, 'inf_tod', 
                        np.where(data['age_years'] <= 6.0, 'preschooler', 'school')))
    data.drop('age_years', axis=1, inplace=True)

    # Create dummies
    data = pd.get_dummies(data, columns=['pupil_right_reaction', 'pupil_left_reaction', 'age_group'])
    data = data.drop(['patid', 'csn'], axis=1)

    # Select columns
    data = data[col_names]

    # Create dataset
    y_val = data['label']
    X_val = data.drop('label', axis=1)
    X_val.columns = col_names_fixed[:-1]
    X_train_non_norm = X.copy()
    X_val_non_norm = X_val.copy()

    # Normalize
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X))
    X_val = pd.DataFrame(scaler.transform(X_val))
    X_train.columns = cols
    X_val.columns = cols

    # Summarize class derivation cohort
    pre_counter = Counter(y)
    pos_weight = pre_counter[0] / pre_counter[1]

    # Balance derivation dataset (undersampling)
    undersampler = RandomUnderSampler(sampling_strategy=0.3, random_state=42)
    X_train, y_train = undersampler.fit_resample(X_train, y)
    X_train_non_norm, _ = undersampler.fit_resample(X_train_non_norm, y)
    post_counter = Counter(y_train)

    return X_train, X_val, y_train, y_val, pos_weight, pre_counter, post_counter, X_train_non_norm, X_val_non_norm
