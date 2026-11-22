"""
Data preprocessing, feature engineering, and stratified splitting pipeline
for Credit Card Fraud Detection.
Prevents data leakage by strictly fitting scalers and resamplers on train split only.
"""
import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

FEATURE_COLUMNS = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
TARGET_COLUMN = 'Class'


def load_raw_dataset(csv_path: str) -> pd.DataFrame:
    """Load dataset from CSV path with integrity checks."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Run dataset/download_dataset.py first."
        )
    df = pd.read_csv(csv_path)
    
    # Check for target column
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset must contain '{TARGET_COLUMN}' column.")
    
    # Drop exact duplicates if present
    initial_len = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dedup_len = len(df)
    if initial_len != dedup_len:
        print(f"Removed {initial_len - dedup_len:,} duplicate transaction rows.")

    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Perform 3-way stratified split: Train (70%), Validation (15%), Test (15%).
    Validation set is used for threshold selection and model comparison.
    Test set is held out strictly for final unbiased evaluation.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # Step 1: Split out Test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Step 2: Split remaining data into Train and Validation
    # Adjust val_size relative to the remaining data
    adjusted_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=adjusted_val_size,
        stratify=y_temp,
        random_state=random_state
    )

    return (
        X_train.reset_index(drop=True),
        X_val.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_val.reset_index(drop=True),
        y_test.reset_index(drop=True)
    )


def fit_scaler(X_train: pd.DataFrame) -> RobustScaler:
    """Fit RobustScaler on 'Time' and 'Amount' features of training data only."""
    scaler = RobustScaler()
    scaler.fit(X_train[['Time', 'Amount']])
    return scaler


def transform_features(X: pd.DataFrame, scaler: RobustScaler) -> pd.DataFrame:
    """Scale 'Time' and 'Amount' features using a pre-fitted scaler."""
    X_scaled = X.copy()
    scaled_cols = scaler.transform(X[['Time', 'Amount']])
    X_scaled['Time'] = scaled_cols[:, 0]
    X_scaled['Amount'] = scaled_cols[:, 1]
    return X_scaled


def apply_smote(
    X_train_scaled: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: float = 0.1,  # 10% minority ratio to prevent overwhelming noise
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE exclusively to the training set.
    Never applied to validation or test sets to ensure realistic evaluation.
    """
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
    return pd.DataFrame(X_resampled, columns=FEATURE_COLUMNS), pd.Series(y_resampled, name=TARGET_COLUMN)


def save_scaler(scaler: RobustScaler, filepath: str) -> None:
    """Save fitted scaler to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(scaler, filepath)


def load_scaler(filepath: str) -> RobustScaler:
    """Load fitted scaler from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fitted scaler not found at {filepath}")
    return joblib.load(filepath)


def preprocess_input_features(
    input_data: pd.DataFrame,
    scaler: Optional[RobustScaler] = None
) -> pd.DataFrame:
    """
    Preprocess raw input transaction features for inference.
    Validates presence of all 30 expected features and transforms Time and Amount.
    """
    # Verify required columns
    missing_cols = [c for c in FEATURE_COLUMNS if c not in input_data.columns]
    if missing_cols:
        raise ValueError(f"Input features missing required columns: {missing_cols}")

    df_features = input_data[FEATURE_COLUMNS].copy()

    # Fill any NaNs if present with median/0
    df_features = df_features.fillna(0.0)

    if scaler is not None:
        df_features = transform_features(df_features, scaler)

    return df_features


def load_and_preprocess_data(
    csv_path: str,
    scaler_save_path: Optional[str] = None,
    apply_smote_train: bool = True,
    sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Full preprocessing pipeline orchestration.
    Returns scaled splits and fitted scaler dictionary.
    """
    raw_df = load_raw_dataset(csv_path)

    if sample_size and sample_size < len(raw_df):
        print(f"Sampling {sample_size:,} rows for rapid development/testing...")
        fraud_df = raw_df[raw_df[TARGET_COLUMN] == 1]
        legit_df = raw_df[raw_df[TARGET_COLUMN] == 0].sample(
            n=sample_size - len(fraud_df), random_state=42
        )
        raw_df = pd.concat([legit_df, fraud_df]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(raw_df)

    # Fit scaler strictly on train split
    scaler = fit_scaler(X_train)
    if scaler_save_path:
        save_scaler(scaler, scaler_save_path)

    # Scale all splits
    X_train_scaled = transform_features(X_train, scaler)
    X_val_scaled = transform_features(X_val, scaler)
    X_test_scaled = transform_features(X_test, scaler)

    # Apply SMOTE to training set only
    if apply_smote_train:
        X_train_final, y_train_final = apply_smote(X_train_scaled, y_train)
    else:
        X_train_final, y_train_final = X_train_scaled, y_train

    return {
        "X_train": X_train_final,
        "y_train": y_train_final,
        "X_val": X_val_scaled,
        "y_val": y_val,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "scaler": scaler,
        "raw_counts": {
            "total": len(raw_df),
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
            "train_resampled": len(X_train_final),
            "fraud_total": int(raw_df[TARGET_COLUMN].sum())
        }
    }
