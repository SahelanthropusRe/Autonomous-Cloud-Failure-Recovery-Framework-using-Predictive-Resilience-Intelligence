
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "prepared_usage.csv"

sys.path.insert(0, str(BASE_DIR))

from recovery.recovery_engine import assess_system


# Page configuration
st.set_page_config(
    page_title="Predictive Cloud Recovery",
    page_icon="C",
    layout="wide",
)

st.title("Predictive Cloud Failure Recovery")
st.caption(
    "Autonomous Recovery Framework using Predictive Resilience Intelligence"
)




# ==================================================
# 1. REAL WORKLOAD DATA ANALYSIS
# ==================================================

st.header("1. Google ClusterData2019 — Workload Analysis")


@st.cache_data
def load_usage(path):
    return pd.read_csv(path)


if DATA_FILE.exists():
    try:
        usage = load_usage(str(DATA_FILE))

        col1, col2, col3 = st.columns(3)

        col1.metric("Usage Records", f"{len(usage):,}")
        col2.metric(
            "Unique Machines",
            f"{usage['machine_id'].nunique():,}",
        )
        col3.metric(
            "Distinct Workloads",
            f"{usage[['collection_id', 'instance_index']].drop_duplicates().shape[0]:,}",
        )

        st.caption(
            "Real workload measurements from Google ClusterData2019. "
            "These statistics are not failure predictions."
        )

        # Resource usage percentiles
        st.subheader("CPU and Memory Usage Distribution")

        resource_data = usage[
            ["average_usage_cpus", "average_usage_memory"]
        ].rename(
            columns={
                "average_usage_cpus": "Average CPU",
                "average_usage_memory": "Average Memory",
            }
        )

        percentiles = resource_data.quantile(
            [0.10, 0.25, 0.50, 0.75, 0.90]
        )

        percentiles.index = [
            "10th percentile",
            "25th percentile",
            "Median",
            "75th percentile",
            "90th percentile",
        ]

        st.line_chart(percentiles)

        st.caption(
            "CPU and memory values are normalized resource measurements, "
            "not percentages."
        )

        # Descriptive statistics
        st.subheader("Resource Statistics")

        summary = resource_data.describe().T.rename(
            columns={
                "count": "Records",
                "mean": "Mean",
                "std": "Std Deviation",
                "min": "Minimum",
                "25%": "25th Percentile",
                "50%": "Median",
                "75%": "75th Percentile",
                "max": "Maximum",
            }
        )

        st.dataframe(
            summary.round(6),
            use_container_width=True,
        )

        # Workload sample
        st.subheader("Sample Workload Records")

        sample_columns = [
            "machine_id",
            "collection_id",
            "instance_index",
            "average_usage_cpus",
            "average_usage_memory",
            "maximum_usage_cpus",
            "maximum_usage_memory",
        ]

        sample = usage[sample_columns].head(100).copy()

        if "start_time_us" in usage.columns:
            sample.insert(
                0,
                "Elapsed Time (hours)",
                (
                    usage["start_time_us"].head(100).to_numpy()
                    / 3_600_000_000
                ).round(2),
            )

        st.dataframe(
            sample,
            use_container_width=True,
            hide_index=True,
        )

    except (ValueError, KeyError, OSError) as exc:
        st.error(f"Unable to load workload data: {exc}")

else:
    st.warning(
        "Prepared dataset not found. Run "
        "python data\\prepare_dataset.py first."
    )


st.divider()


# ==================================================
# 2. SIMULATED RECOVERY ENGINE
# ==================================================

st.header("2. Simulated Recovery Demonstration")

st.sidebar.header("System Telemetry")
st.sidebar.caption("Adjust the simulated metrics below.")

cpu = st.sidebar.slider("CPU Usage (%)", 0, 100, 96)
memory = st.sidebar.slider("Memory Usage (%)", 0, 100, 94)
network = st.sidebar.slider("Network Usage (%)", 0, 100, 90)
latency = st.sidebar.slider("Latency (ms)", 0, 1000, 480)
errors = st.sidebar.slider("Error Rate (%)", 0.0, 20.0, 9.0, 0.5)

metrics = {
    "cpu_usage": float(cpu),
    "memory_usage": float(memory),
    "network_usage": float(network),
    "latency_ms": float(latency),
    "error_rate": float(errors),
}

try:
    result = assess_system(metrics)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Model Risk Score",
        f"{result['risk_probability']:.1%}",
    )
    col2.metric("Risk Level", result["risk_level"])
    col3.metric("Execution Mode", "Simulation Only")

    st.progress(result["risk_probability"])

    st.subheader("Recovery Recommendation")
    st.write(f"**Recommended Action:** {result['recommended_action']}")
    st.write(result["explanation"])

    st.caption(
        "Thresholds are illustrative demonstration rules, not "
        "empirically validated operational limits."
    )

    # Abnormal metrics
    st.subheader("Metrics Exceeding Demo Thresholds")

    abnormal = result.get("abnormal_metrics", [])

    if abnormal:
        st.dataframe(
            pd.DataFrame(abnormal),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("No metrics exceed the configured demo thresholds.")

    # Input summary
    st.subheader("Current Simulated Telemetry")

    telemetry = pd.DataFrame(
        {
            "Metric": [
                "CPU Usage",
                "Memory Usage",
                "Network Usage",
                "Latency",
                "Error Rate",
            ],
            "Value": [
                f"{cpu}%",
                f"{memory}%",
                f"{network}%",
                f"{latency} ms",
                f"{errors:.1f}%",
            ],
        }
    )

    st.dataframe(
        telemetry,
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Technical Details"):
        st.json(result)

        st.write(
            "Execution mode: SIMULATION_ONLY. No infrastructure is "
            "modified. Real-data failure prediction is not yet validated."
        )

except (FileNotFoundError, ValueError, KeyError) as exc:
    st.error(f"Recovery assessment failed: {exc}")


# Footer
st.divider()
st.caption(
    "Academic Research Prototype | Local Execution | "
    "Real Workload Analysis | Simulated Recovery"
)
