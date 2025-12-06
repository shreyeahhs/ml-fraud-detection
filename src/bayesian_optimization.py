import json
import os

import optuna
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.feature_engineering import prepare_training_data


def objective(trial: optuna.Trial):
    """
    Objective function for Optuna.

    For each trial:
    - Suggest a set of XGBoost hyperparameters.
    - Train a model.
    - Return validation ROC-AUC.
    """
    X_train, y_train, X_valid, y_valid = prepare_training_data()

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        **params,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    valid_pred_proba = model.predict_proba(X_valid)[:, 1]
    roc_auc = roc_auc_score(y_valid, valid_pred_proba)

    return roc_auc


def run_bayesian_optimization(
    n_trials: int = 30,
    models_dir: str = "models"
):
    """
    Run Optuna study to optimize XGBoost hyperparameters.

    Parameters
    ----------
    n_trials : int
        Number of hyperparameter combinations to try.
    models_dir : str
        Where to save best params and the Optuna study.
    """
    os.makedirs(models_dir, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        study_name="xgb_fraud_opt",
        storage=f"sqlite:///{os.path.join(models_dir, 'bayes_opt_study.db')}",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=n_trials)

    print("Best value (ROC-AUC):", study.best_value)
    print("Best params:", study.best_params)

    # Save best params to JSON so train.py can use them if desired
    best_params_path = os.path.join(models_dir, "best_xgb_params.json")
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=4)

    print(f"Saved best params to {best_params_path}")


if __name__ == "__main__":
    run_bayesian_optimization(n_trials=30)

