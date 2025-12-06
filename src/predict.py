import json
import os
from typing import List, Dict, Union

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.feature_engineering import (
    add_handcrafted_features,
    apply_label_encoders,
    apply_scaler,
    load_feature_artifacts,
)


def load_model(models_dir: str = "models") -> XGBClassifier:
    """
    Load the trained XGBoost model from disk.
    """
    model_path = os.path.join(models_dir, "xgb_model.json")
    model = XGBClassifier()
    model.load_model(model_path)
    return model


def preprocess_input_dataframe(
    df: pd.DataFrame,
    models_dir: str = "models"
) -> pd.DataFrame:
    """
    Apply the same feature engineering steps for inference.

    - Add handcrafted features.
    - Load encoders and scaler.
    - Encode categoricals.
    - Scale numeric features.
    - Reorder / align columns to exactly match the
      training feature order saved in feature_config.
    """
    encoders, scaler, feature_config = load_feature_artifacts(models_dir)

    numeric_features = feature_config["numeric_features"]
    categorical_features = feature_config["categorical_features"]
    # This is the exact column order used during training
    all_features = feature_config.get(
        "all_features",
        numeric_features + categorical_features  # fallback for old configs
    )

    # 1. Add handcrafted features
    df = add_handcrafted_features(df)

    # 2. Apply label encoders to categoricals
    df = apply_label_encoders(df, encoders)

    # 3. Ensure all numeric features exist before scaling,
    #    but add them in a single concat to avoid fragmentation
    missing_numeric = [col for col in numeric_features if col not in df.columns]
    if missing_numeric:
        numeric_zeros = pd.DataFrame(
            0.0, index=df.index, columns=missing_numeric
        )
        df = pd.concat([df, numeric_zeros], axis=1)

    # 4. Scale numeric features
    df = apply_scaler(df, numeric_features, scaler)

    # 5. Finally, reindex to the full training feature list.
    #    This will:
    #      - Add any missing columns (numeric or categorical) with 0.
    #      - Drop any extra columns sent in the request.
    #      - Put columns in the exact same order as training.
    df = df.reindex(columns=all_features, fill_value=0)

    return df



def predict_proba(
    input_data: Union[pd.DataFrame, List[Dict]],
    models_dir: str = "models"
) -> np.ndarray:
    """
    Predict fraud probability for a DataFrame or a list of dicts.

    Parameters
    ----------
    input_data : DataFrame or list of dict
        New data to score.
    models_dir : str
        Folder where model and artifacts are stored.

    Returns
    -------
    numpy array of shape (n_samples,)
        Probability of fraud for each row.
    """
    if isinstance(input_data, list):
        df = pd.DataFrame(input_data)
    else:
        df = input_data.copy()

    model = load_model(models_dir=models_dir)
    X = preprocess_input_dataframe(df, models_dir=models_dir)

    proba = model.predict_proba(X)[:, 1]
    return proba


def predict_labels(
    input_data: Union[pd.DataFrame, List[Dict]],
    threshold: float = 0.5,
    models_dir: str = "models"
) -> np.ndarray:
    """
    Predict binary fraud labels based on a threshold.

    Why thresholds:
    --------------
    - The model outputs probabilities.
    - We must choose a threshold above which we call a transaction "fraud".
    - Threshold choice trades off false positives vs false negatives.

    Parameters
    ----------
    input_data : DataFrame or list of dict
    threshold : float
        Cutoff for predicting fraud.
    models_dir : str

    Returns
    -------
    numpy array of shape (n_samples,)
        1 for fraud, 0 for non-fraud.
    """
    proba = predict_proba(input_data, models_dir=models_dir)
    labels = (proba >= threshold).astype(int)
    return labels


if __name__ == "__main__":
    # Small manual test
    sample = [
        {"TransactionAmt": 100.0, "card1": 1000, "ProductCD": "W"},
        {"TransactionAmt": 500.0, "card1": 2000, "ProductCD": "C"},
    ]
    probs = predict_proba(sample)
    print("Predicted probabilities:", probs)
