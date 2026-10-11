
from pathlib import Path
import json

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
METRICS_FILE = ROOT / "reports" / "metrics.json"
DRIFT_FILE = ROOT / "reports" / "drift_report.csv"
REFERENCE_FILE = ROOT / "data" / "reference.csv"
PRODUCTION_FILE = ROOT / "data" / "production.csv"

st.set_page_config(
    page_title="ML Model Monitor",
    page_icon="📊",
    layout="wide",
)

st.title("ML Model Monitoring Platform")
st.caption("Model performance • Data quality • Feature drift")

# Verify that the previous project steps generated their files.
required_files = [
    METRICS_FILE,
    DRIFT_FILE,
    REFERENCE_FILE,
    PRODUCTION_FILE,
]

missing = [str(path.relative_to(ROOT)) for path in required_files if not path.exists()]

if missing:
    st.error("Required project files are missing:")
    for filename in missing:
        st.write(f"- {filename}")
    st.info("Run `python train.py` and then `python monitor.py shifted`.")
    st.stop()

with METRICS_FILE.open(encoding="utf-8") as file:
    metrics = json.load(file)

drift = pd.read_csv(DRIFT_FILE)
reference = pd.read_csv(REFERENCE_FILE)
production = pd.read_csv(PRODUCTION_FILE)

drift["drift_detected"] = (
    drift["drift_detected"].astype(str).str.lower().eq("true")
)

drift_count = int(drift["drift_detected"].sum())
feature_count = len(drift)

# Header metrics
st.subheader("Monitoring overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Model accuracy", f"{metrics['accuracy']:.1%}")
c2.metric("Model F1 score", f"{metrics['f1']:.1%}")
c3.metric("Features monitored", feature_count)
c4.metric("Features flagged", drift_count)

if drift_count:
    st.warning(
        f"{drift_count} of {feature_count} features were flagged "
        "for possible data drift in the latest report."
    )
else:
    st.success("No features were flagged in the latest report.")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Reference vs production")
    st.write(f"Reference rows: **{len(reference)}**")
    st.write(f"Production rows: **{len(production)}**")
    st.write(f"Input features: **{reference.shape[1]}**")
    st.caption(
        "Production data is simulated for this portfolio project."
    )

with right:
    st.subheader("Model evaluation")
    st.write(f"Precision: **{metrics['precision']:.1%}**")
    st.write(f"Recall: **{metrics['recall']:.1%}**")
    st.write(f"Training rows: **{metrics['training_rows']}**")
    st.write(f"Test rows: **{metrics['test_rows']}**")

st.divider()

st.subheader("Feature drift analysis")

chart_data = (
    drift[["feature", "ks_statistic"]]
    .dropna()
    .set_index("feature")
    .sort_values("ks_statistic", ascending=False)
    .head(10)
)


# Render the drift chart without Streamlit's Altair dependency.
for feature, row in chart_data.iterrows():
    score = float(row["ks_statistic"])
    width = max(0, min(100, score * 100))
    st.markdown(
        f"""
        <div style="margin-bottom:12px">
          <div style="display:flex;justify-content:space-between;
                      margin-bottom:4px">
            <span>{feature}</span>
            <strong>{score:.3f}</strong>
          </div>
          <div style="background:#e5e7eb;border-radius:5px;height:12px">
            <div style="background:#e58b35;width:{width:.2f}%;
                        height:12px;border-radius:5px"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.caption(
    "Higher KS statistics indicate larger differences between "
    "the reference and production feature distributions."
)

st.subheader("Feature-level results")

status_filter = st.selectbox(
    "Filter results",
    ["All features", "Drift only", "No drift flagged"],
)

if status_filter == "Drift only":
    visible = drift[drift["drift_detected"]]
elif status_filter == "No drift flagged":
    visible = drift[~drift["drift_detected"]]
else:
    visible = drift

st.dataframe(
    visible.sort_values("p_value", na_position="last"),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download drift report",
    data=drift.to_csv(index=False),
    file_name="drift_report.csv",
    mime="text/csv",
)

st.caption(
    "A feature drift alert does not by itself prove that model "
    "performance has declined. Evaluate production labels separately."
)
