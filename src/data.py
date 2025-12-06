import pandas as pd

df = pd.read_parquet("data/processed/train_processed.parquet")  # or your file

reference = df.sample(10000, random_state=42)
current   = df.sample(5000, random_state=123)

reference.to_csv("data/reference_scored.csv", index=False)
current.to_csv("data/current_scored.csv", index=False)
