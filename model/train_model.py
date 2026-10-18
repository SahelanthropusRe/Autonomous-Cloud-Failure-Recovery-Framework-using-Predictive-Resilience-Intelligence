import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "telemetry.csv"
MODEL_PATH = BASE_DIR / "models" / "failure_model.joblib"

FEATURES = [
    "cpu_usage",
    "memory_usage",
    "network_usage",
    "latency_ms",
    "error_rate"
]

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Telemetry data not found: {DATA_PATH}\n"
            "Run: python data\\generate_telemetry.py"
        )

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df["failure_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions,
        target_names=["Normal", "High Risk"],
        zero_division=0
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": FEATURES},
        MODEL_PATH
    )
    print(f"\nModel saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
