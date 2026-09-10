import os
import json
import joblib
import numpy as np

# Load trained Random Forest model artifact
MODEL_PATH = os.path.join(os.path.dirname(__file__), "failure_model.joblib")
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    print(f"Warning: {MODEL_PATH} not found. Running in fallback mode.")

STEP_FUNCTION_ARN = "arn:aws:states:us-east-1:123456789012:stateMachine:CloudFailureRecoveryEngine"
RISK_THRESHOLD = 0.70

def predict_failure_probability(cpu_util, mem_util, cpu_slope, mem_slope, scheduling_class=1):
    """
    Runs real inference using the trained Random Forest model.
    Returns: float (probability between 0.0 and 1.0)
    """
    if model is not None:
        features = np.array([[cpu_util, mem_util, cpu_slope, mem_slope, scheduling_class]])
        # Probability of class 1 (failure)
        prob = model.predict_proba(features)[0][1]
        return round(float(prob), 4)
    
    # Fallback heuristic if joblib file is missing
    score = 0.0
    if mem_util > 0.85: score += 0.4
    if cpu_util > 0.88: score += 0.3
    if mem_slope > 0.03: score += 0.3
    return round(min(score, 1.0), 4)

def lambda_handler(event, context):
    """
    AWS Lambda entrypoint triggered by metrics stream or HTTP event.
    """
    body = json.loads(event.get("body", "{}")) if isinstance(event, dict) and "body" in event else event
    if not isinstance(body, dict):
        body = {}

    instance_id = body.get("instance_id", "i-mock-instance-01")
    cpu_util = float(body.get("cpu_util", 0.50))
    mem_util = float(body.get("mem_util", 0.50))
    cpu_slope = float(body.get("cpu_slope", 0.0))
    mem_slope = float(body.get("mem_slope", 0.0))
    scheduling_class = int(body.get("scheduling_class", 1))

    # Real ML Inference
    risk_score = predict_failure_probability(
        cpu_util=cpu_util,
        mem_util=mem_util,
        cpu_slope=cpu_slope,
        mem_slope=mem_slope,
        scheduling_class=scheduling_class
    )

    action_triggered = risk_score >= RISK_THRESHOLD

    result = {
        "instance_id": instance_id,
        "risk_score": risk_score,
        "action_triggered": action_triggered,
        "status": "RECOVERY_TRIGGERED" if action_triggered else "STABLE"
    }

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(result)
    }

if __name__ == "__main__":
    # Test 1: Normal healthy workload
    normal_payload = {
        "instance_id": "i-test-healthy",
        "cpu_util": 0.25,
        "mem_util": 0.40,
        "cpu_slope": 0.001,
        "mem_slope": 0.002,
        "scheduling_class": 1
    }
    print("Normal test:", lambda_handler(normal_payload, None))

    # Test 2: Failing condition (high mem + steep upward trend)
    failing_payload = {
        "instance_id": "i-test-exhaustion",
        "cpu_util": 0.88,
        "mem_util": 0.94,
        "cpu_slope": 0.04,
        "mem_slope": 0.05,
        "scheduling_class": 2
    }
    print("Failing test:", lambda_handler(failing_payload, None))