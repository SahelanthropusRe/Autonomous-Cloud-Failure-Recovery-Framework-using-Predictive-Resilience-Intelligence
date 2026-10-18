from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "instance_usage_sample.parquet.gz"
OUTPUT = ROOT / "prepared_usage.csv"


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Dataset not found: {SOURCE}")

    print("Reading Google ClusterData2019...")
    parquet = pq.ParquetFile(SOURCE)

    frames = []

    columns = [
        "start_time",
        "end_time",
        "machine_id",
        "collection_id",
        "instance_index",
        "average_usage",
        "maximum_usage",
    ]

    for batch in parquet.iter_batches(
        batch_size=50000,
        columns=columns,
    ):
        frame = batch.to_pandas()

        for usage_type in ["average_usage", "maximum_usage"]:
            for resource in ["cpus", "memory"]:
                frame[f"{usage_type}_{resource}"] = frame[
                    usage_type
                ].apply(
                    lambda value: (
                        value.get(resource)
                        if isinstance(value, dict)
                        else None
                    )
                )

        frame = frame.drop(
            columns=["average_usage", "maximum_usage"]
        )
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)

    data = data.rename(
        columns={
            "start_time": "start_time_us",
            "end_time": "end_time_us",
        }
    )

    data["start_time"] = pd.to_datetime(
        data["start_time_us"],
        unit="us",
        errors="coerce",
    )

    data["end_time"] = pd.to_datetime(
        data["end_time_us"],
        unit="us",
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            "machine_id",
            "start_time",
            "average_usage_cpus",
            "average_usage_memory",
        ]
    )

    data = data.sort_values(
        ["machine_id", "start_time"]
    ).reset_index(drop=True)

    data.to_csv(OUTPUT, index=False)

    print("\nDataset preparation complete.")
    print(f"Rows prepared: {len(data):,}")
    print(f"Unique machines: {data['machine_id'].nunique():,}")
    print(f"Output file: {OUTPUT}")
    print("\nAvailable columns:")
    print(", ".join(data.columns))


if __name__ == "__main__":
    main()
