
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent

for folder in ("models", "data", "reports"):
    (ROOT / folder).mkdir(parents=True, exist_ok=True)

# Load a built-in dataset; no download is needed.
dataset = load_breast_cancer(as_frame=True)
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    ),
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

metrics = {
    "dataset": "Breast Cancer Wisconsin",
    "training_rows": len(X_train),
    "test_rows": len(X_test),
    "features": X.shape[1],
    "accuracy": accuracy_score(y_test, predictions),
    "precision": precision_score(
        y_test, predictions, zero_division=0
    ),
    "recall": recall_score(
        y_test, predictions, zero_division=0
    ),
    "f1": f1_score(y_test, predictions, zero_division=0),
}

joblib.dump(model, ROOT / "models" / "model.joblib")
X_train.to_csv(ROOT / "data" / "reference.csv", index=False)

with (ROOT / "reports" / "metrics.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(metrics, file, indent=2)

print("MODEL TRAINING SUCCESSFUL")
print(json.dumps(metrics, indent=2))
print("Saved model: models/model.joblib")
print("Saved reference data: data/reference.csv")
print("Saved metrics: reports/metrics.json")

