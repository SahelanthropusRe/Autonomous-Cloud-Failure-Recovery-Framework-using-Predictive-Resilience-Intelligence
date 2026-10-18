
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

USAGE_FILE = DATA / "prepared_usage.csv"
EVENT_FILE = DATA / "instance_events_sample.parquet.gz"
OUTPUT = DATA / "training_data.csv"

HORIZON_US = 5 * 60 * 1_000_000


def main():
    print("Loading usage records...")
    usage = pd.read_csv(USAGE_FILE)

    print("Loading valid terminal events...")
    events = pq.read_table(
        EVENT_FILE,
        columns=[
            "time",
            "type",
            "collection_id",
            "instance_index",
        ],
    ).to_pandas()

    # Remove missing and sentinel timestamps.
    events["time"] = pd.to_numeric(
        events["time"], errors="coerce"
    )
    events = events[
        events["time"].between(0, 3_000_000_000_000)
    ].copy()

    events = events[
        events["type"].isin([5, 6])
    ].dropna(
        subset=["collection_id", "instance_index"]
    ).copy()

    keys = ["collection_id", "instance_index"]

    for frame in (usage, events):
        for key in keys:
            frame[key] = pd.to_numeric(
                frame[key], errors="coerce"
            )

    usage = usage.dropna(
        subset=keys + ["start_time_us", "end_time_us"]
    ).copy()
    events = events.dropna(
        subset=keys + ["time"]
    ).copy()

    for frame in (usage, events):
        for key in keys:
            frame[key] = frame[key].astype("int64")

    events["time"] = events["time"].astype("int64")

    # This is an initial instance-level approximation:
    # select the earliest observed FAIL or FINISH event.
    terminal = (
        events.sort_values("time")
        .drop_duplicates(subset=keys, keep="first")
        .rename(
            columns={
                "time": "terminal_time_us",
                "type": "terminal_type",
            }
        )
    )

    print("Joining usage to terminal events...")
    data = usage.merge(
        terminal,
        on=keys,
        how="inner",
        validate="many_to_one",
    )

    # Keep only measurements preceding the event,
    # with the event occurring within five minutes.
    data["time_to_event_us"] = (
        data["terminal_time_us"] - data["end_time_us"]
    )

    data = data[
        data["time_to_event_us"].between(0, HORIZON_US)
    ].copy()

    features = [
        "average_usage_cpus",
        "average_usage_memory",
        "maximum_usage_cpus",
        "maximum_usage_memory",
    ]

    data[features] = data[features].apply(
        pd.to_numeric, errors="coerce"
    )
    data = data.dropna(subset=features)

    data["failure_label"] = (
        data["terminal_type"] == 5
    ).astype("int8")

    data.to_csv(OUTPUT, index=False)

    print("\nTraining-data preparation finished.")
    print(f"Rows retained: {len(data):,}")
    print("Label counts:")
    print(
        data["failure_label"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    print(f"Output: {OUTPUT}")
    print(
        "\nThis is a preliminary dataset. Validate sample size, "
        "event matching, and temporal leakage before training."
    )


if __name__ == "__main__":
    main()
