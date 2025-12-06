import json
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def get_feature_lists(df: pd.DataFrame, target_col: str = "isFraud") -> Tuple[List[str], List[str]]:
    """
    Separate numeric and categorical columns.

    Parameters
    ----------
    df : DataFrame
        Input data.
    target_col : str
        Name of the target column.

    Returns
    -------
    numeric_features : list of str
        Names of numeric feature columns.
    categorical_features : list of str
        Names of categorical feature columns.
    """
    numeric_features = []
    categorical_features = []

    for col in df.columns:
        if col == target_col:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    return numeric_features, categorical_features


def add_handcrafted_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add new features that may help the model.

    Examples:
    - log of transaction amount
    - day of week from TransactionDT if available
    - some simple ratios

    Parameters
    ----------
    df : DataFrame
        Input data.

    Returns
    -------
    DataFrame
        Data with new columns added.
    """
    # Log transaction amount (avoid log(0) by adding a small value)
    if "TransactionAmt" in df.columns:
        df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])

    # Example: extract hour from a time delta column if present
    # In this dataset, TransactionDT is a time delta from a reference point.
    # We can create a "day" or "hour" like feature if needed.
    if "TransactionDT" in df.columns:
        df["Transaction_day"] = (df["TransactionDT"] // (60 * 60 * 24)).astype(int)
        df["Transaction_hour"] = (df["TransactionDT"] // (60 * 60) % 24).astype(int)

    return df


def fit_label_encoders(
    df: pd.DataFrame,
    categorical_features: List[str]
) -> Dict[str, LabelEncoder]:
    """
    Fit a LabelEncoder for each categorical column.

    We store one encoder per column so we can apply
    the same mapping during inference.

    Parameters
    ----------
    df : DataFrame
        Training data.
    categorical_features : list of str
        Names of categorical columns.

    Returns
    -------
    encoders : dict
        Mapping from column name to fitted LabelEncoder.
    """
    encoders: Dict[str, LabelEncoder] = {}

    for col in categorical_features:
        le = LabelEncoder()
        # Fill missing values with a special token
        df[col] = df[col].fillna("missing")
        le.fit(df[col].astype(str))
        encoders[col] = le

    return encoders


def apply_label_encoders(
    df: pd.DataFrame,
    encoders: Dict[str, LabelEncoder]
) -> pd.DataFrame:
    """
    Apply fitted label encoders to a DataFrame.

    For categories not seen during training, we use a fallback value.
    One simple approach is to map unseen values to -1.

    Parameters
    ----------
    df : DataFrame
        Data to transform.
    encoders : dict
        Mapping from column name to LabelEncoder.

    Returns
    -------
    DataFrame
        Data with encoded categorical columns.
    """
    for col, le in encoders.items():
        if col not in df.columns:
            continue

        df[col] = df[col].fillna("missing").astype(str)

        # Map known classes to integers
        known_classes = set(le.classes_)
        df[col] = df[col].apply(
            lambda x: x if x in known_classes else "unknown_category"
        )

        # If "unknown_category" is new, we extend the classes
        if "unknown_category" not in le.classes_:
            le.classes_ = np.append(le.classes_, "unknown_category")

        df[col] = le.transform(df[col])

    return df


def fit_scaler(
    df: pd.DataFrame,
    numeric_features: List[str]
) -> StandardScaler:
    """
    Fit StandardScaler to numeric columns.

    Why we scale numeric features:
    ------------------------------
    - It helps some models converge better.
    - It makes numeric values more comparable.
    - XGBoost is less sensitive than linear models, but scaling
      can still help when we mix features with very different scales.

    Parameters
    ----------
    df : DataFrame
        Training data.
    numeric_features : list of str
        Names of numeric columns.

    Returns
    -------
    scaler : StandardScaler
        Fitted scaler.
    """
    scaler = StandardScaler()
    scaler.fit(df[numeric_features].fillna(0.0))
    return scaler


def apply_scaler(
    df: pd.DataFrame,
    numeric_features: List[str],
    scaler: StandardScaler
) -> pd.DataFrame:
    """
    Apply StandardScaler to numeric columns.

    Parameters
    ----------
    df : DataFrame
        Data to transform.
    numeric_features : list of str
        Names of numeric columns.
    scaler : StandardScaler
        Fitted scaler.

    Returns
    -------
    DataFrame
        Data with scaled numeric columns.
    """
    df[numeric_features] = scaler.transform(df[numeric_features].fillna(0.0))
    return df


def save_feature_artifacts(
    encoders: Dict[str, LabelEncoder],
    scaler: StandardScaler,
    numeric_features: List[str],
    categorical_features: List[str],
    all_features: List[str],
    models_dir: str = "models"
) -> None:
    """
    Save encoders, scaler, and feature configuration to disk.

    We now also save `all_features`, which is the exact
    column order used when training the model. We will
    reuse this order during inference so XGBoost sees
    exactly the same feature_names.
    """
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(encoders, os.path.join(models_dir, "label_encoders.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

    feature_config = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "all_features": all_features,  # NEW: training column order
    }

    with open(os.path.join(models_dir, "feature_config.json"), "w") as f:
        json.dump(feature_config, f)


def load_feature_artifacts(
    models_dir: str = "models"
) -> Tuple[Dict[str, LabelEncoder], StandardScaler, Dict[str, List[str]]]:
    """
    Load encoders, scaler, and feature configuration from disk.
    """
    encoders = joblib.load(os.path.join(models_dir, "label_encoders.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))

    with open(os.path.join(models_dir, "feature_config.json"), "r") as f:
        feature_config = json.load(f)

    return encoders, scaler, feature_config


def prepare_training_data(
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    target_col: str = "isFraud"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Full feature engineering pipeline for training:

    1. Load processed train and valid data.
    2. Add handcrafted features.
    3. Detect numeric and categorical columns.
    4. Fit label encoders and scaler on train.
    5. Transform both train and valid.
    6. Save encoders and feature config.

    Returns
    -------
    X_train, y_train, X_valid, y_valid
    """
    train_path = os.path.join(processed_dir, "train_processed.parquet")
    valid_path = os.path.join(processed_dir, "valid_processed.parquet")

    train_df = pd.read_parquet(train_path)
    valid_df = pd.read_parquet(valid_path)

    # Add handcrafted features
    train_df = add_handcrafted_features(train_df)
    valid_df = add_handcrafted_features(valid_df)

    # Separate target
    y_train = train_df[target_col]
    y_valid = valid_df[target_col]

    train_df = train_df.drop(columns=[target_col])
    valid_df = valid_df.drop(columns=[target_col])

    # Get feature lists
    numeric_features, categorical_features = get_feature_lists(
        pd.concat([train_df, valid_df], axis=0)
    )

    # Fit encoders on train
    encoders = fit_label_encoders(train_df, categorical_features)
    train_df = apply_label_encoders(train_df, encoders)
    valid_df = apply_label_encoders(valid_df, encoders)

    # Fit scaler on train
    scaler = fit_scaler(train_df, numeric_features)
    train_df = apply_scaler(train_df, numeric_features, scaler)
    valid_df = apply_scaler(valid_df, numeric_features, scaler)

    # After encoding and scaling, capture the exact training column order
    all_features = list(train_df.columns)

    # Save artifacts, including the full ordered feature list
    save_feature_artifacts(
        encoders,
        scaler,
        numeric_features,
        categorical_features,
        all_features=all_features,
        models_dir=models_dir,
    )

    return train_df, y_train, valid_df, y_valid
    


def prepare_test_data(
    processed_dir: str = "data/processed",
    models_dir: str = "models"
) -> pd.DataFrame:
    """
    Apply the same feature engineering steps to the test data.

    We must use the encoders and scaler fitted on train,
    otherwise the model will see inconsistent values.

    Returns
    -------
    X_test : DataFrame
        Test data ready for inference.
    """
    test_path = os.path.join(processed_dir, "test_processed.parquet")
    test_df = pd.read_parquet(test_path)

    # Add handcrafted features
    test_df = add_handcrafted_features(test_df)

    # Load artifacts
    encoders, scaler, feature_config = load_feature_artifacts(models_dir)

    numeric_features = feature_config["numeric_features"]
    categorical_features = feature_config["categorical_features"]

    # Apply encoders and scaler
    test_df = apply_label_encoders(test_df, encoders)
    test_df = apply_scaler(test_df, numeric_features, scaler)

    # Ensure columns order matches training
    all_features = numeric_features + categorical_features
    # Some columns might be missing in test, fill them with zeros
    for col in all_features:
        if col not in test_df.columns:
            test_df[col] = 0

    test_df = test_df[all_features]

    return test_df


if __name__ == "__main__":
    # Simple manual test of the pipeline
    X_train, y_train, X_valid, y_valid = prepare_training_data()
    print("Train shape:", X_train.shape)
    print("Valid shape:", X_valid.shape)
