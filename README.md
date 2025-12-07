# Fraud Detection Using Kaggle Credit Card Transactions Dataset

**Author:** Shreyas Gowda
**Program:** MSc Data Science
**Institution:** University of Glasgow
**Project:** Fraud Detection (Machine Learning)

---

##  Table of Contents

1. [Project Overview](#project-overview)
2. [Objectives](#objectives)
3. [Dataset Description](#dataset-description)
4. [Directory Structure](#directory-structure)
5. [Technologies & Libraries](#technologies--libraries)
6. [Methodology](#methodology)
7. [Models Implemented](#models-implemented)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Results](#results)
10. [How to Run the Project](#how-to-run-the-project)
11. [Challenges & Solutions](#challenges--solutions)
12. [Future Work](#future-work)
13. [References & Credits](#references--credits)
14. [License](#license)
15. [Acknowledgements](#acknowledgements)

---

##  Project Overview

This project focuses on developing a robust **fraud detection system** using machine learning techniques applied to the **Kaggle Credit Card Fraud Detection** dataset.
Due to the extremely imbalanced nature of the data (fraudulent cases are less than 0.2%), the project explores a variety of strategies including resampling techniques, anomaly detection models, and cost-sensitive learning.

The aim is to produce a model that is:

* Highly sensitive to fraudulent behaviour
* Interpretable wherever possible
* Scalable and reproducible
* Well-documented for academic and industrial use

---

##  Objectives

* Perform deep exploratory data analysis (EDA) to understand behavioural patterns.
* Handle severe class imbalance effectively using advanced resampling methods.
* Train a variety of machine learning models and compare performance.
* Build an evaluation pipeline tailored for fraud detection.
* Document every stage for reproducibility and clarity.

---

##  Dataset Description

**Source:** Kaggle — *Credit Card Fraud Detection*
**Link:** Search "Credit Card Fraud Detection" on Kaggle (due to licensing, link excluded)

### Dataset Summary

* **Total rows:** 284,807
* **Fraud cases:** 492
* **Imbalance ratio:** ~1:577
* **Features:**

  * **Time:** Seconds elapsed between each transaction and the first recorded transaction
  * **Amount:** Transaction amount
  * **V1–V28:** Principal Component Analysis (PCA) transformed features protecting confidentiality
  * **Class:**

    * 0 → Legitimate
    * 1 → Fraud

### Preprocessing Performed

* Missing values verification (dataset contains no nulls)
* Scaling of `Amount` and `Time` using StandardScaler
* Train-test splitting using stratification
* Oversampling using **SMOTE**, **ADASYN**, and hybrid techniques
* Undersampling using **NearMiss** variants

---

##  Directory Structure

```
project/
│── data/
│   ├── creditcard.csv
│── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Modelling.ipynb
│   └── 04_Evaluation.ipynb
│── src/
│   ├── utils.py
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
│── reports/
│   ├── results_table.csv
│   └── model_comparisons.png
│── README.md
│── requirements.txt
│── LICENSE
```

---

##  Technologies & Libraries

* **Python 3.10+**
* **NumPy** / **Pandas**
* **Matplotlib** / **Seaborn** for visualizations
* **Scikit-learn** for ML models
* **Imbalanced-Learn** for resampling techniques
* **XGBoost**, **LightGBM**, **CatBoost** for gradient boosted modelling
* **TensorFlow / Keras** (optional) for deep models

---

##  Methodology

### 1️ Exploratory Data Analysis (EDA)

* Distribution analysis (legitimate vs fraudulent)
* Correlation heatmaps
* Box plots to detect variance patterns
* Time-based patterns
* Transaction amount frequency analysis

### 2️ Data Preprocessing

* Standardization of numerical features
* Handling imbalance with:

  * SMOTE
  * SMOTEENN
  * Random undersampling
  * Ensemble-based balancing (BalancedRandomForest)

### 3️ Feature Selection

* PCA components are already anonymised, so interpretability is limited
* Importance was derived from:

  * Feature importance from tree-based models
  * Permutation importance
  * SHAP values

### 4️ Model Training

* Logistic Regression (baseline)
* Random Forest
* XGBoost
* LightGBM
* CatBoost
* Isolation Forest (anomaly detection)
* Autoencoders (deep learning anomaly detection)

### 5️ Evaluation

Given imbalance, the main metrics:

* **Precision**
* **Recall** (critical)
* **F1 Score**
* **ROC AUC**
* **PR AUC** (more informative for imbalanced tasks)
* Confusion matrix analysis

---

##  Models Implemented

| Model               | Type                | Purpose                               |
| ------------------- | ------------------- | ------------------------------------- |
| Logistic Regression | Baseline classifier | Benchmark comparison                  |
| Random Forest       | Ensemble            | Captures non-linear patterns          |
| XGBoost             | Gradient Boosting   | High performance and interpretability |
| LightGBM            | Gradient Boosting   | Fast training on large datasets       |
| CatBoost            | Gradient Boosting   | Handles categorical data well         |
| Isolation Forest    | Anomaly Detection   | Fraud = rare event                    |
| Autoencoder         | Deep Learning       | Learns reconstruction errors          |

---

##  Results (Example Summary)

> Replace these with your actual values

* **Best Model:** XGBoost
* **Recall:** ~0.93
* **Precision:** ~0.85
* **F1 Score:** ~0.89
* **PR AUC:** ~0.95

Confusion matrix and ROC curves are included in `/reports/`.

---

##  How to Run the Project

### Step 1 — Clone Repository

```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Step 2 — Install Dependencies

```
pip install -r requirements.txt
```

### Step 3 — Run Notebooks

Open `notebooks/` using Jupyter or VS Code.

### Step 4 — Run Scripts

```
python src/models.py
```

---

##  Challenges & Solutions

### Severe Class Imbalance

**Challenge:** Fraud cases < 0.2%
**Solution:** Hybrid resampling + anomaly detection + custom threshold tuning.

### Interpretability

**Challenge:** PCA-transformed features
**Solution:** Tree-based feature importance + SHAP analysis.

### Data Leakage

**Challenge:** Balancing only the training set
**Solution:** Strict use of stratification and pipeline-based balancing.

---

##  Future Work

* Deploying model via FastAPI or Flask
* Real-time anomaly detection pipeline
* Integration with streaming platforms (Kafka)
* Explainability dashboards using SHAP or LIME
* Federated learning for privacy-preserving fraud detection

---

##  References & Credits

* **Dataset:** Credit Card Fraud Detection — Kaggle
* **Scikit-learn Documentation:** [https://scikit-learn.org](https://scikit-learn.org)
* **Imbalanced-learn Documentation:** [https://imbalanced-learn.org](https://imbalanced-learn.org)
* Research papers on fraud detection & imbalance handling
* Community forums and discussions on Kaggle & StackOverflow
* **GPT assistance used for generating documentation structure and code comments**

---

##  License

This project is licensed under the **MIT License**.
Feel free to reuse and modify with attribution.

---

##  Acknowledgements

Thanks to open-source contributors, Kaggle dataset providers, and the machine learning research community whose work makes projects like this possible.

---

##  Made with ❤️ by **Shreyas Gowda**

…and documented with a little help from **GPT** for better understanding and readability of code.
