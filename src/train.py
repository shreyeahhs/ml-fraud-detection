import os

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.feature_engineering import prepare_training_data


def train_xgb_model(
    models_dir: str = "models",
    processed_dir: str = "data/processed"
) -> None:
    """
    Train an XGBoost model on the IEEE-CIS fraud dataset.

    Steps:
    1. Load and prepare training and validation data.
    2. Define an XGBoost classifier with reasonable defaults.
    3. Train the model.
    4. Evaluate ROC-AUC on validation set.
    5. Save the trained model.

    Parameters
    ----------
    models_dir : str
        Where to save the trained model.
    processed_dir : str
        Where processed data lives.
    """
    print("Preparing training and validation data...")
    X_train, y_train, X_valid, y_valid = prepare_training_data(
        processed_dir=processed_dir,
        models_dir=models_dir,
    )

    print("Defining XGBoost model...")
    model = XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        reg_lambda=1.0,
        reg_alpha=0.0,
    )

    print("Training model...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=50,
    )

    print("Evaluating on validation set...")
    valid_pred_proba = model.predict_proba(X_valid)[:, 1]
    roc_auc = roc_auc_score(y_valid, valid_pred_proba)
    print(f"Validation ROC-AUC: {roc_auc:.4f}")

    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "xgb_model.json")

    print(f"Saving model to {model_path}...")
    model.save_model(model_path)

    # Save validation predictions for monitoring examples
    valid_pred_path = os.path.join(models_dir, "valid_predictions.npy")
    np.save(valid_pred_path, valid_pred_proba)

    print("Training complete.")


if __name__ == "__main__":
    train_xgb_model()
