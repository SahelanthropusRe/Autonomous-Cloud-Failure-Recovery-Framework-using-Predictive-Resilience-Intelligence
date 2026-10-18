
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "training_data.csv"
MODEL_FILE = ROOT / "models" / "real_failure_model.joblib"

FEATURES = [
    "average_usage_cpus",
    "average_usage_memory",
    "maximum_usage_cpus",
    "maximum_usage_memory",
]


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Training data not found: {DATA_FILE}"
        )

    data = pd.read_csv(DATA_FILE)

    if "failure_label" not in data.columns:
        raise ValueError("Missing failure_label column.")

    data = data.dropna(subset=FEATURES + ["failure_label"])
    data = data[data["failure_label"].isin([0, 1])]

    print(f"Available records: {len(data):,}")
    print("Class distribution:")
    print(data["failure_label"].value_counts().sort_index())

    if len(data) < 100 or data["failure_label"].nunique() < 2:
        raise ValueError(
            "Not enough usable records from both classes. "
            "Do not train or report model performance yet."
        )

    X = data[FEATURES]
    y = data["failure_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print(f"\nTraining records: {len(X_train):,}")
    print(f"Testing records: {len(X_test):,}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["FINISH", "FAIL"],
            zero_division=0,
        )
    )
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    if y_test.nunique() == 2:
        print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    else:
        print("ROC-AUC unavailable: test set has only one class.")

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "dataset": "Google ClusterData2019",
            "label_definition": (
                "Preliminary FAIL versus FINISH terminal-event label"
            ),
        },
        MODEL_FILE,
    )

    print(f"\nSaved real-data model: {MODEL_FILE}")


if __name__ == "__main__":
    main()
