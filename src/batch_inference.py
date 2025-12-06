import argparse
import os

import numpy as np
import pandas as pd

from src.predict import predict_proba, predict_labels


def run_batch_inference(
    input_csv: str,
    output_csv: str,
    threshold: float = 0.5
) -> None:
    """
    Run batch inference on a CSV file.

    Steps:
    ------
    1. Read input CSV into a DataFrame.
    2. Call the prediction functions.
    3. Attach predictions to the DataFrame.
    4. Save to output CSV.

    Why batch inference:
    --------------------
    In many practical settings, new transactions are collected over time.
    For example, every hour you might run a job that scores all transactions
    created in that hour. This is batch inference, and it is more efficient
    and easier to manage than scoring every transaction one by one.
    """
    print(f"Reading input data from {input_csv}")
    df = pd.read_csv(input_csv)

    print("Running predictions...")
    probabilities = predict_proba(df)
    labels = predict_labels(df, threshold=threshold)

    df["fraud_probability"] = probabilities
    df["fraud_label"] = labels

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    print(f"Writing results to {output_csv}")
    df.to_csv(output_csv, index=False)
    print("Batch inference complete.")


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Batch inference for fraud detection")
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input CSV with new transactions",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to output CSV with predictions",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for predicting fraud label",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_batch_inference(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        threshold=args.threshold,
    )
