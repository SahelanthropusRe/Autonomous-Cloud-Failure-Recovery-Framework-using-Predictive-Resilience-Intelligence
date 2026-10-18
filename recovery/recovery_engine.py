
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "failure_model.joblib"

# These thresholds are illustrative operational rules for the demo,
# not thresholds learned from Google ClusterData2019.
METRIC_THRESHOLDS = {
    "cpu_usage": ("CPU usage", 80),
    "memory_usage": ("Memory usage", 85),
    "network_usage": ("Network usage", 85),
    "latency_ms": ("Latency", 300),
    "error_rate": ("Error rate", 5),
}


def assess_system(metrics):
    """Assess simulated telemetry and recommend a simulated recovery action."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run: python models\\train_model.py"
        )

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    features = bundle["features"]

    missing = [key for key in features if key not in metrics]
    if missing:
        raise ValueError(f"Missing metrics: {missing}")

    for key in features:
        value = metrics[key]

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value < 0
        ):
            raise ValueError(
                f"Metric '{key}' must be a non-negative number."
            )

    X = pd.DataFrame(
        [[metrics[key] for key in features]],
        columns=features,
    )

    probability = float(model.predict_proba(X)[0][1])

    if probability >= 0.80:
        level = "CRITICAL"
        action = "SIMULATED_FAILOVER"
        explanation = (
            "The demo model reports high risk. "
            "Recommend moving the workload to a healthy instance."
        )
    elif probability >= 0.60:
        level = "HIGH"
        action = "SIMULATED_RESTART"
        explanation = (
            "The demo model reports elevated risk. "
            "Recommend investigating and simulating a service restart."
        )
    elif probability >= 0.35:
        level = "ELEVATED"
        action = "INCREASE_MONITORING"
        explanation = (
            "The demo model reports moderate risk. "
            "Recommend closer monitoring and investigation."
        )
    else:
        level = "NORMAL"
        action = "CONTINUE_MONITORING"
        explanation = (
            "The demo model reports lower risk. "
            "Continue monitoring; this does not guarantee system health."
        )

    abnormal_metrics = []

    for key, (label, threshold) in METRIC_THRESHOLDS.items():
        if key in metrics and metrics[key] >= threshold:
            abnormal_metrics.append({
                "metric": label,
                "value": metrics[key],
                "threshold": threshold,
            })

    if abnormal_metrics:
        explanation += (
            f" {len(abnormal_metrics)} input metric(s) exceed "
            "the demo's configured thresholds."
        )
    else:
        explanation += " No input metric exceeds the configured thresholds."

    return {
        "risk_probability": round(probability, 4),
        "risk_level": level,
        "recommended_action": action,
        "explanation": explanation,
        "abnormal_metrics": abnormal_metrics,
        "execution_mode": "SIMULATION_ONLY",
        "model_type": "Synthetic-data Random Forest",
        "thresholds_are_demo_rules": True,
    }


if __name__ == "__main__":
    sample = {
        "cpu_usage": 96,
        "memory_usage": 94,
        "network_usage": 90,
        "latency_ms": 480,
        "error_rate": 9,
    }

    result = assess_system(sample)

    print("SIMULATED CLOUD RECOVERY ASSESSMENT")
    print("-" * 40)

    for key, value in result.items():
        print(f"{key}: {value}")
