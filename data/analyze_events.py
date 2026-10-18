from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).resolve().parent
SOURCE = DATA_DIR / "instance_events_sample.parquet.gz"
OUTPUT = DATA_DIR / "event_type_summary.csv"

EVENT_TYPES = {
    0: "SUBMIT",
    1: "QUEUE",
    2: "ENABLE",
    3: "SCHEDULE",
    4: "EVICT",
    5: "FAIL",
    6: "FINISH",
    7: "KILL",
    8: "LOST",
    9: "UPDATE_PENDING",
    10: "UPDATE_RUNNING",
}


def main():
    print("Reading real instance events...")
    table = pq.read_table(SOURCE, columns=["time", "type"])

    events = table.to_pandas()
    events["event_name"] = events["type"].map(EVENT_TYPES)
    events["is_failure"] = events["type"].eq(5)

    summary = (
        events.groupby(["type", "event_name"], dropna=False)
        .agg(
            event_count=("type", "size"),
            failure_count=("is_failure", "sum"),
        )
        .reset_index()
        .sort_values("type")
    )

    summary.to_csv(OUTPUT, index=False)

    print("\nEvent summary:")
    print(summary.to_string(index=False))
    print(f"\nTotal events: {len(events):,}")
    print(f"FAIL events: {events['is_failure'].sum():,}")
    print(f"Saved summary: {OUTPUT}")


if __name__ == "__main__":
    main()
