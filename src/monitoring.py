import os
from typing import Tuple, List

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency


def load_reference_and_current(
    reference_path: str,
    current_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load reference and current datasets from CSV.

    Parameters
    ----------
    reference_path : str
        Path to the historical / training data.
    current_path : str
        Path to the recent / production data.

    Returns
    -------
    (reference, current) : Tuple[pd.DataFrame, pd.DataFrame]
    """
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)
    return reference, current


def is_categorical(series: pd.Series, max_unique_ratio: float = 0.2) -> bool:
    """
    Decide if a column should be treated as categorical.

    Logic (simple, not perfect):
    - If dtype is 'object' or 'category' -> categorical.
    - Otherwise, if unique values are relatively few compared to total
      rows (e.g., ratio < max_unique_ratio), we also treat as categorical.

    Parameters
    ----------
    series : pd.Series
    max_unique_ratio : float
        Maximum ratio of unique values / total length to consider as categorical.

    Returns
    -------
    bool
    """
    if series.dtype == "O" or str(series.dtype).startswith("category"):
        return True

    # For numeric low-cardinality columns (e.g., codes 0/1/2), treat as categorical.
    unique_count = series.nunique(dropna=True)
    total = len(series)
    if total == 0:
        return False
    return (unique_count / total) <= max_unique_ratio


def run_ks_test(
    ref: pd.Series,
    cur: pd.Series
) -> Tuple[float, float]:
    """
    Run Kolmogorov–Smirnov test for numeric drift.

    Parameters
    ----------
    ref : pd.Series
        Reference numeric values.
    cur : pd.Series
        Current numeric values.

    Returns
    -------
    statistic : float
    p_value : float
    """
    # Drop NaNs for the test
    ref_clean = ref.dropna()
    cur_clean = cur.dropna()

    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return np.nan, np.nan

    stat, p_value = ks_2samp(ref_clean, cur_clean)
    return stat, p_value


def run_chi_square(
    ref: pd.Series,
    cur: pd.Series
) -> Tuple[float, float]:
    """
    Run Chi-square test for categorical drift.

    We build a contingency table of category counts in reference and current,
    then apply chi-square.

    Parameters
    ----------
    ref : pd.Series
        Reference categorical values.
    cur : pd.Series
        Current categorical values.

    Returns
    -------
    statistic : float
    p_value : float
    """
    ref_counts = ref.value_counts(dropna=False)
    cur_counts = cur.value_counts(dropna=False)

    # Convert category labels to strings to avoid mixed-type sorting issues
    ref_map = {str(k): int(v) for k, v in ref_counts.items()}
    cur_map = {str(k): int(v) for k, v in cur_counts.items()}

    # Union of all category labels (as strings)
    all_categories = sorted(set(ref_map.keys()).union(set(cur_map.keys())))

    # Build aligned count arrays in the same order
    ref_aligned = np.array([ref_map.get(cat, 0) for cat in all_categories], dtype=int)
    cur_aligned = np.array([cur_map.get(cat, 0) for cat in all_categories], dtype=int)

    contingency = np.vstack([ref_aligned, cur_aligned])

    # If everything is zero or too trivial, return NaNs
    if contingency.sum() == 0 or contingency.shape[1] == 0:
        return np.nan, np.nan

    stat, p_value, _, _ = chi2_contingency(contingency)
    return stat, p_value


def analyze_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target_col: str = "isFraud",
    prediction_col: str = "fraud_probability",
    p_value_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Analyze drift for each common feature between reference and current.

    We ignore target and prediction columns because we mainly care about
    input drift.

    Parameters
    ----------
    reference : DataFrame
    current : DataFrame
    target_col : str
    prediction_col : str
    p_value_threshold : float
        Threshold below which we say "drift detected".

    Returns
    -------
    summary_df : DataFrame
        One row per feature with:
        - column
        - type (numeric/categorical)
        - test (ks/chi-square/skipped)
        - statistic
        - p_value
        - drift (True/False/Unknown)
        - ref_mean / cur_mean (for numeric)
        - ref_top / cur_top (for categorical)
    """
    # Use only columns present in both
    common_columns = [c for c in reference.columns if c in current.columns]

    # Remove target/prediction if present
    ignore_cols: List[str] = [target_col, prediction_col]
    features = [c for c in common_columns if c not in ignore_cols]

    rows = []

    for col in features:
        ref_col = reference[col]
        cur_col = current[col]

        # Decide numeric vs categorical
        if is_categorical(ref_col):
            col_type = "categorical"
            test_name = "chi-square"

            stat, p_value = run_chi_square(ref_col, cur_col)

            # Top category info for interpretability
            ref_top = str(ref_col.value_counts(dropna=False).head(1).index.tolist()[0]) if ref_col.notna().any() else "NA"
            cur_top = str(cur_col.value_counts(dropna=False).head(1).index.tolist()[0]) if cur_col.notna().any() else "NA"

            ref_mean = np.nan
            cur_mean = np.nan

        else:
            col_type = "numeric"
            test_name = "ks"
            stat, p_value = run_ks_test(ref_col, cur_col)

            ref_top = ""
            cur_top = ""
            ref_mean = float(ref_col.mean()) if ref_col.notna().any() else np.nan
            cur_mean = float(cur_col.mean()) if cur_col.notna().any() else np.nan

        if np.isnan(stat) or np.isnan(p_value):
            drift = "Unknown"
        else:
            drift = bool(p_value < p_value_threshold)

        rows.append(
            {
                "column": col,
                "type": col_type,
                "test": test_name,
                "statistic": stat,
                "p_value": p_value,
                "drift": drift,
                "ref_mean": ref_mean,
                "cur_mean": cur_mean,
                "ref_top": ref_top,
                "cur_top": cur_top,
            }
        )

    summary_df = pd.DataFrame(rows)
    return summary_df


def generate_html_report(
    summary_df: pd.DataFrame,
    output_html: str,
    reference_path: str,
    current_path: str,
    p_value_threshold: float = 0.05,
) -> None:
    """
    Generate a simple HTML report from the drift summary DataFrame.

    The HTML is intentionally simple:
    - A short explanation in plain English.
    - A table with statistics per feature.

    Parameters
    ----------
    summary_df : DataFrame
    output_html : str
    reference_path : str
    current_path : str
    p_value_threshold : float
    """
    os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)

    n_features = len(summary_df)
    n_drifted = (summary_df["drift"] == True).sum()

    # Basic HTML with inline CSS for readability
    html = []
    html.append("<html><head><meta charset='utf-8'><title>Data Drift Report</title>")
    html.append(
        """
        <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2, h3 { font-family: Arial, sans-serif; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ccc; padding: 6px 8px; font-size: 12px; }
        th { background-color: #f5f5f5; }
        tr.drift-true { background-color: #ffe5e5; }
        tr.drift-false { background-color: #e8f5e9; }
        tr.drift-unknown { background-color: #eeeeee; }
        </style>
        """
    )
    html.append("</head><body>")

    html.append("<h1>Data Drift Report (Custom, No Evidently)</h1>")
    html.append("<p>This report compares a <b>reference dataset</b> and a <b>current dataset</b> to detect feature-level drift.</p>")
    html.append(f"<p><b>Reference file:</b> {reference_path}<br>")
    html.append(f"<b>Current file:</b> {current_path}</p>")

    html.append("<h2>Summary</h2>")
    html.append(f"<p>Total features checked: <b>{n_features}</b><br>")
    html.append(f"Features with drift (p &lt; {p_value_threshold}): <b>{n_drifted}</b></p>")

    html.append("<h2>How to read this report (in simple English)</h2>")
    html.append(
        f"""
        <p>
        For each feature, we compare the values in the reference data and the
        current data using a statistical test:
        </p>
        <ul>
            <li>Numeric features: Kolmogorov–Smirnov (KS) test.</li>
            <li>Categorical features: Chi-square test.</li>
        </ul>
        <p>
        The <b>p-value</b> tells us how likely it is that the two distributions
        (reference vs current) are actually the same.
        If the p-value is below <b>{p_value_threshold}</b>, we say there is
        <b>drift</b> for that feature.
        </p>
        """
    )

    html.append("<h2>Feature-level details</h2>")
    html.append("<table>")
    html.append(
        "<tr>"
        "<th>Column</th>"
        "<th>Type</th>"
        "<th>Test</th>"
        "<th>Statistic</th>"
        "<th>p-value</th>"
        "<th>Drift?</th>"
        "<th>Ref mean</th>"
        "<th>Cur mean</th>"
        "<th>Ref top category</th>"
        "<th>Cur top category</th>"
        "</tr>"
    )

    for _, row in summary_df.iterrows():
        drift = row["drift"]
        if drift is True:
            tr_class = "drift-true"
            drift_label = "Yes"
        elif drift is False:
            tr_class = "drift-false"
            drift_label = "No"
        else:
            tr_class = "drift-unknown"
            drift_label = "Unknown"

        html.append(f"<tr class='{tr_class}'>")
        html.append(f"<td>{row['column']}</td>")
        html.append(f"<td>{row['type']}</td>")
        html.append(f"<td>{row['test']}</td>")
        html.append(f"<td>{row['statistic']:.4f}" if not pd.isna(row['statistic']) else "<td>NA</td>")
        html.append(f"<td>{row['p_value']:.4f}" if not pd.isna(row['p_value']) else "<td>NA</td>")
        html.append(f"<td>{drift_label}</td>")
        html.append(f"<td>{'' if pd.isna(row['ref_mean']) else round(row['ref_mean'], 4)}</td>")
        html.append(f"<td>{'' if pd.isna(row['cur_mean']) else round(row['cur_mean'], 4)}</td>")
        html.append(f"<td>{row['ref_top']}</td>")
        html.append(f"<td>{row['cur_top']}</td>")
        html.append("</tr>")

    html.append("</table>")
    html.append("</body></html>")

    with open(output_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    print(f"Custom data drift report saved to {output_html}")


def main() -> None:
    """
    Entry point for running monitoring from the command line.

    Default paths:
        data/reference_scored.csv
        data/current_scored.csv

    You can create these by:
    - Taking a sample of your training/validation data as reference.
    - Taking a recent batch of production-scored data as current.
    """
    ref_path = "data/reference_scored.csv"
    curr_path = "data/current_scored.csv"
    output_html = "models/data_drift_report.html"

    if not os.path.exists(ref_path) or not os.path.exists(curr_path):
        print(
            "Reference or current scored CSV not found.\n"
            f"  Expected:\n"
            f"    {ref_path}\n"
            f"    {curr_path}\n"
            "Please create them first (for example by sampling your processed data)."
        )
        return

    reference_df, current_df = load_reference_and_current(ref_path, curr_path)
    summary_df = analyze_drift(reference_df, current_df)
    generate_html_report(summary_df, output_html, ref_path, curr_path)


if __name__ == "__main__":
    main()
