
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"

REFERENCE_PATH = DATA_DIR / "reference.csv"
PRODUCTION_PATH = DATA_DIR / "production.csv"
REPORT_PATH = REPORT_DIR / "drift_report.csv"

# Usage: python monitor.py normal
#        python monitor.py shifted
scenario = sys.argv[1].lower() if len(sys.argv) > 1 else "shifted"

if scenario not in {"normal", "shifted"}:
    raise SystemExit("Choose scenario: normal or shifted")

if not REFERENCE_PATH.exists():
    raise SystemExit("Missing data/reference.csv. Run python train.py first.")

reference = pd.read_csv(REFERENCE_PATH)

if len(reference) < 2:
    raise SystemExit("Reference data must contain at least two rows.")

# Use a reproducible sample to simulate an incoming data batch.
rng = np.random.default_rng(42)
production = reference.sample(
    n=300,
    replace=True,
    random_state=42,
).reset_index(drop=True)

# Simulate changes in the first three feature distributions.
# This is synthetic test data, not actual production data.
if scenario == "shifted":
    for column in reference.columns[:3]:
        std = reference[column].std()
        if pd.notna(std) and std > 0:
            production[column] = production[column] + 1.5 * std

DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
production.to_csv(PRODUCTION_PATH, index=False)

results = []

for column in reference.columns:
    ref_values = reference[column].dropna()
    prod_values = production[column].dropna()

    if ref_values.empty or prod_values.empty:
        results.append({
            "feature": column,
            "reference_rows": len(ref_values),
            "production_rows": len(prod_values),
            "ks_statistic": np.nan,
            "p_value": np.nan,
            "drift_detected": False,
            "status": "INSUFFICIENT_DATA",
        })
        continue

    statistic, p_value = ks_2samp(ref_values, prod_values)
    drift = bool(p_value < 0.05)

    results.append({
        "feature": column,
        "reference_rows": len(ref_values),
        "production_rows": len(prod_values),
        "ks_statistic": float(statistic),
        "p_value": float(p_value),
        "drift_detected": drift,
        "status": "DRIFT" if drift else "OK",
    })

report = pd.DataFrame(results)
report.to_csv(REPORT_PATH, index=False)

drift_count = int(report["drift_detected"].sum())

print(f"\nMONITORING SCENARIO: {scenario.upper()}")
print(f"Reference rows: {len(reference)}")
print(f"Production rows: {len(production)}")
print(f"Features checked: {len(report)}")
print(f"Features flagged: {drift_count}")
print("\nTop feature results:")
print(
    report.sort_values("p_value", na_position="last")
    .head(10)
    .to_string(index=False)
)
print(f"\nProduction data saved: {PRODUCTION_PATH}")
print(f"Drift report saved: {REPORT_PATH}")
