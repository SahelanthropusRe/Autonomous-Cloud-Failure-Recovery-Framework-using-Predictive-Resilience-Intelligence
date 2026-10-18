import numpy as np
import pandas as pd
from pathlib import Path

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
