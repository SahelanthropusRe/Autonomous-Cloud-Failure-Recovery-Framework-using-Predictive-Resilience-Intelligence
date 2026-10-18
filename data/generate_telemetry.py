import numpy as np
import pandas as pd
from pathlib import Path

def generate_telemetry(n_samples=3000, seed=42):
    rng = np.random.default_rng(seed)

    cpu = rng.uniform(10, 95, n_samples)
    memory = rng.uniform(20, 95, n_samples)
    network = rng.uniform(5, 100, n_samples)
    latency = rng.uniform(20, 500, n_samples)
    error_rate = rng.uniform(0, 10, n_samples)

    risk_score = (
        0.30 * (cpu / 100)
        + 0.25 * (memory / 100)
        + 0.15 * (network / 100)
        + 0.15 * (latency / 500)
        + 0.15 * (error_rate / 10)
    )

   
    failure_risk = (
        (risk_score > 0.63)
        | ((cpu > 88) & (memory > 80))
        | ((latency > 400) & (error_rate > 6))
    ).astype(int)

    df = pd.DataFrame({
        "cpu_usage": cpu.round(2),
        "memory_usage": memory.round(2),
        "network_usage": network.round(2),
        "latency_ms": latency.round(2),
        "error_rate": error_rate.round(2),
        "risk_score": risk_score.round(4),
        "failure_risk": failure_risk
    })

    return df

if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = generate_telemetry()
    output_file = output_dir / "telemetry.csv"
    df.to_csv(output_file, index=False)

    print(f"Generated {len(df)} telemetry records.")
    print(f"High-risk records: {df['failure_risk'].sum()}")
    print(f"Saved to: {output_file}")
    print("\nSample records:")
    print(df.head())
    print("\nRisk distribution:")
    print(df["failure_risk"].value_counts())
