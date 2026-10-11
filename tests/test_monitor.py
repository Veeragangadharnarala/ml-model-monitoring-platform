
from pathlib import Path
import json

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_model_exists_and_loads():
    model_path = ROOT / "models" / "model.joblib"
    assert model_path.exists()
    model = joblib.load(model_path)
    assert hasattr(model, "predict")


def test_reference_data_is_valid():
    path = ROOT / "data" / "reference.csv"
    assert path.exists()

    reference = pd.read_csv(path)
    assert len(reference) > 0
    assert reference.shape[1] == 30
    assert reference.notna().all().all()


def test_metrics_are_valid():
    path = ROOT / "reports" / "metrics.json"
    assert path.exists()

    with path.open(encoding="utf-8") as file:
        metrics = json.load(file)

    for name in ("accuracy", "precision", "recall", "f1"):
        assert 0 <= metrics[name] <= 1


def test_drift_report_is_valid():
    path = ROOT / "reports" / "drift_report.csv"
    assert path.exists()

    report = pd.read_csv(path)
    assert len(report) == 30
    assert {
        "feature",
        "p_value",
        "drift_detected",
        "status",
    }.issubset(report.columns)
