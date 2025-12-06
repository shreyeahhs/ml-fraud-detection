import os
from typing import Tuple, List

import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw_data(
    data_dir: str = "data/raw"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the raw CSV files from the Kaggle IEEE-CIS dataset.

    We separate them into:
    - train_transaction
    - train_identity
    - test_transaction
    - test_identity

    Parameters
    ----------
    data_dir : str
        Path to the folder that contains the raw CSV files.

    Returns
    -------
    Tuple of four DataFrames.
    """
    train_transaction_path = os.path.join(data_dir, "train_transaction.csv")
    train_identity_path = os.path.join(data_dir, "train_identity.csv")
    test_transaction_path = os.path.join(data_dir, "test_transaction.csv")
    test_identity_path = os.path.join(data_dir, "test_identity.csv")

    train_transaction = pd.read_csv(train_transaction_path)
    train_identity = pd.read_csv(train_identity_path)
    test_transaction = pd.read_csv(test_transaction_path)
    test_identity = pd.read_csv(test_identity_path)

    return train_transaction, train_identity, test_transaction, test_identity


def merge_datasets(
    transaction: pd.DataFrame,
    identity: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge transaction and identity tables on TransactionID.

    Why we merge datasets:
    ----------------------
    The IEEE-CIS dataset splits information into two tables:
    - transaction: information about the transaction itself
    - identity: additional information about the device and user

    Many useful features for fraud detection live in both tables.
    If we kept them separate, the model would not see all available
    information for each transaction.

    So we merge them on the shared key column, TransactionID, to create
    one big table per split (train and test).

    Parameters
    ----------
    transaction : DataFrame
        Transaction data with target column 'isFraud' in train.
    identity : DataFrame
        Identity data with device and user information.

    Returns
    -------
    DataFrame
        Merged table with columns from both inputs.
    """
    merged = transaction.merge(identity, how="left", on="TransactionID")
    return merged


def basic_cleaning(df: pd.DataFrame, drop_cols: List[str] = None) -> pd.DataFrame:
    """
    Perform simple cleaning steps.

    We keep this function very basic:
    - Drop obvious identifier columns that should not be used as features.
    - Leave missing values as they are. They will be handled later.

    Parameters
    ----------
    df : DataFrame
        Input data.
    drop_cols : list of str
        Columns to drop. If None, use default list.

    Returns
    -------
    DataFrame
        Cleaned DataFrame.
    """
    if drop_cols is None:
        drop_cols = ["TransactionID"]  # we usually drop this as it is just an ID

    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return df


def split_train_valid(
    df: pd.DataFrame,
    target_col: str = "isFraud",
    valid_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into training and validation sets.

    Why we split:
    -------------
    We want to train on one part of the data (train set)
    and evaluate on another part (validation set).
    The validation set simulates unseen data.

    Parameters
    ----------
    df : DataFrame
        Full dataset with target column.
    target_col : str
        Name of the target column (fraud label).
    valid_size : float
        Fraction of rows to keep in the validation set.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    train_df, valid_df : DataFrame, DataFrame
        The split DataFrames.
    """
    train_df, valid_df = train_test_split(
        df,
        test_size=valid_size,
        random_state=random_state,
        stratify=df[target_col],
    )
    return train_df, valid_df


def save_processed_data(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    processed_dir: str = "data/processed"
) -> None:
    """
    Save processed DataFrames to disk.

    We use Parquet format because:
    - It is efficient.
    - It keeps column types.
    - It is widely supported.

    Parameters
    ----------
    train_df : DataFrame
        Training set.
    valid_df : DataFrame
        Validation set.
    test_df : DataFrame
        Test set.
    processed_dir : str
        Folder to save processed files.
    """
    os.makedirs(processed_dir, exist_ok=True)

    train_df.to_parquet(os.path.join(processed_dir, "train_processed.parquet"))
    valid_df.to_parquet(os.path.join(processed_dir, "valid_processed.parquet"))
    test_df.to_parquet(os.path.join(processed_dir, "test_processed.parquet"))


def main():
    """
    End-to-end preprocessing pipeline:
    1. Load raw data.
    2. Merge transaction and identity tables.
    3. Basic cleaning.
    4. Split train into train and valid.
    5. Save processed files.
    """
    print("Loading raw data...")
    train_transaction, train_identity, test_transaction, test_identity = load_raw_data()

    print("Merging train tables...")
    train_merged = merge_datasets(train_transaction, train_identity)

    print("Merging test tables...")
    test_merged = merge_datasets(test_transaction, test_identity)

    print("Applying basic cleaning...")
    train_clean = basic_cleaning(train_merged)
    test_clean = basic_cleaning(test_merged)

    print("Splitting into train and validation...")
    train_df, valid_df = split_train_valid(train_clean, target_col="isFraud")

    print("Saving processed data...")
    save_processed_data(train_df, valid_df, test_clean)

    print("Done preprocessing.")


if __name__ == "__main__":
    main()
