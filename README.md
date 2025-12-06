# IEEE-CIS Fraud Detection — End-to-End Machine Learning Project

This repository contains a complete, beginner-friendly, end-to-end machine learning project built using the **IEEE-CIS Fraud Detection Dataset (Kaggle)**.  
It takes you from **raw data → preprocessing → feature engineering → model training → hyperparameter tuning → batch inference → REST API → monitoring with Evidently**.

Everything is written in **simple English**, with clear explanations of *why* each step is necessary.

---

## ⭐ Project Overview

The goal is to build a fraud detection pipeline that:

- Preprocesses & merges raw Kaggle CSV files  
- Performs exploratory data analysis (EDA)  
- Creates new useful features  
- Trains an XGBoost model  
- Tunes hyperparameters using Bayesian Optimization  
- Serves predictions through a FastAPI REST API  
- Supports batch inference for CSV files  
- Monitors data drift in production using Evidently  

---

## 📂 Folder Structure

```
ml-fraud-detection/
├── README.md
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_bayesian_optimization.ipynb
│   └── 05_inference_testing.ipynb
├── src/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── bayesian_optimization.py
│   ├── predict.py
│   ├── fastapi_app.py
│   ├── batch_inference.py
│   └── monitoring.py
├── models/
├── Dockerfile
├── requirements.txt
└── .gitignore
```

---

## 📘 Dataset Description

The dataset contains **two main tables**:

- **transaction** files → transaction-level details  
- **identity** files → device & user-level metadata  

They must be **merged on `TransactionID`** to form a complete picture of each transaction.

### Why merging matters

Fraud behavior often depends on the combination of:

- transaction amount + device fingerprints  
- card details + IP address  
- time of transaction + browser metadata  

Keeping the tables separate would hide these relationships.

---

## 🧹 Preprocessing Steps

### Done in `src/preprocess.py`

- Load raw CSV files  
- Merge datasets  
- Drop useless identifier columns  
- Split into train/validation sets  
- Save processed parquet files  

---

## 🛠 Feature Engineering

Done in `src/feature_engineering.py`.

Includes:

- Label encoding categorical features  
- Scaling numeric features  
- Adding handcrafted features:
  - `TransactionAmt_log`
  - `Transaction_day`
  - `Transaction_hour`

### Why feature engineering improves accuracy

It gives the model **cleaner, more informative** inputs, helping it detect subtle fraud patterns.

---

## 🤖 Model Training (XGBoost)

XGBoost is chosen because:

- Works extremely well on tabular data  
- Handles missing values  
- Captures non-linear relationships  
- Widely used in real-world fraud systems  

Metric used: **ROC-AUC** (good for imbalanced datasets).

Training is implemented in `src/train.py`.

---

## 🔍 Bayesian Optimization

Optuna is used to tune hyperparameters using **Bayesian optimization**, which:

- Learns which combinations of parameters work well  
- Is faster and smarter than grid/random search  

Configured in `src/bayesian_optimization.py`.

---

## 🚀 FastAPI REST API

Run:

```bash
uvicorn src.fastapi_app:app --reload
```

API endpoints:

### `GET /`
Health check.

### `POST /predict`
Predict fraud probability for one or many transactions.

Example request:

```json
{
  "transactions": [
    {"TransactionAmt": 100, "ProductCD": "W", "card1": 1000}
  ]
}
```

Example response:

```json
{
  "predictions": [
    {"probability": 0.12, "label": 0}
  ]
}
```

---

## 📦 Docker Support

Build:

```bash
docker build -t fraud-api .
```

Run:

```bash
docker run -p 8000:8000 fraud-api
```

---

## 📊 Batch Inference

Script: `src/batch_inference.py`

Run:

```bash
python -m src.batch_inference --input_csv data/new.csv --output_csv data/scored.csv
```

Adds:

- `fraud_probability`
- `fraud_label`

---

## 🔎 Monitoring with Evidently

`src/monitoring.py` generates an **HTML drift report**.

Why drift matters:

- Fraud patterns evolve  
- Models degrade over time  
- Drift detection tells you when retraining is needed  

Output example:

```
models/data_drift_report.html
```

---

## 🧪 Jupyter Notebooks

1. **01_eda.ipynb** – Explore dataset  
2. **02_feature_engineering.ipynb** – Build features  
3. **03_model_training.ipynb** – Train & evaluate model  
4. **04_bayesian_optimization.ipynb** – Tune hyperparameters  
5. **05_inference_testing.ipynb** – Validate inference pipeline  

---

## ▶ How to Run the Entire Pipeline

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Kaggle dataset → place in `data/raw/`

### 3. Run preprocessing

```bash
python -m src.preprocess
```

### 4. Train model

```bash
python -m src.train
```

### 5. Start API

```bash
uvicorn src.fastapi_app:app --reload
```

---

## 📸 Placeholder Screenshots

```
docs/images/eda.png
docs/images/feature_importance.png
docs/images/evidently_report.png
docs/images/api_docs.png
```

(Add your own screenshots here.)

---

## ✔ Summary

This project demonstrates a **full ML production workflow**:

- Data preprocessing  
- Feature engineering  
- Model training  
- Hyperparameter tuning  
- Serving predictions  
- Batch scoring  
- Monitoring with Evidently  

It can be used as a template for:

- Fraud detection systems  
- Tabular ML projects  
- Real-world ML pipelines  

---

Happy building!
