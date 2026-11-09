"""Interactive portfolio dashboard. Load generated holdout predictions only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bank Marketing Decision Lab", page_icon="📊", layout="wide")
st.title("Bank Marketing Decision Lab")
st.caption("Predictive decision support · Time-ordered evaluation · Open, reproducible research")

with st.sidebar:
    st.header("Experiment")
    result_dir = Path(st.text_input("Results directory", "reports/demo"))
    capacity = st.slider("Contact capacity (%)", 5, 100, 20, 5)
    value_per_subscription = st.number_input("Illustrative gross value per subscription", min_value=0.0, value=100.0, step=10.0)
    contact_cost = st.number_input("Illustrative cost per call", min_value=0.0, value=5.0, step=1.0)
    st.caption("Financial inputs are user assumptions—not observed campaign economics.")

metrics_path = result_dir / "metrics.json"
preds_path = result_dir / "holdout_predictions.csv"
if not metrics_path.exists() or not preds_path.exists():
    st.info("No experiment output found. Run `banklab demo`, or download the UCI data and run `banklab train`.")
    st.code("pip install -e '.[app,dev]'\nbanklab demo\nstreamlit run app/streamlit_app.py", language="bash")
    st.stop()

manifest = json.loads(metrics_path.read_text(encoding="utf-8"))
frame = pd.read_csv(preds_path)
if manifest.get("synthetic_demo"):
    st.error("SYNTHETIC DEMO — results below are generated from artificial data, not empirical findings.")
else:
    st.info("Historical UCI observations (2008–2010); descriptive predictions, not causal campaign uplift.")

truth = frame["observed_subscribed"].astype(int).to_numpy()
probs = frame["p_champion"].to_numpy()
from banklab.metrics import ranking_at_capacity, illustrative_scenario
selection = ranking_at_capacity(truth, probs, capacity / 100)
scenario = illustrative_scenario(selection, value_per_subscription, contact_cost)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Test observations", f"{len(frame):,}")
c2.metric("Observed subscription base rate", f"{truth.mean():.1%}")
c3.metric(f"Precision at top {capacity}%", f"{selection['precision_at_k']:.1%}")
c4.metric("Lift vs. random", f"{selection['lift_at_k']:.2f}×" if selection['lift_at_k'] is not None else "n/a")

st.subheader("Model evaluation")
champion = manifest["champion"]
st.write(f"Selected model: **{champion.replace('_', ' ').title()}**, chosen using validation average precision.")
cols = st.columns(2)
with cols[0]:
    st.image(str(result_dir / "precision_recall.png"), caption="Final chronological holdout")
with cols[1]:
    st.image(str(result_dir / "targeting_lift.png"), caption="Retrospective targeting lift")

st.subheader("Capacity and scenario")
st.write(f"Contacting **{selection['contacts']:,}** of {len(frame):,} observed records includes "
         f"**{selection['observed_subscriptions']:,} historical subscriptions** versus "
         f"**{selection['random_expected_subscriptions']:.1f}** expected under random selection.")
a, b, c = st.columns(3)
a.metric("Hypothetical gross value", f"{scenario['assumed_gross_value']:,.0f}")
b.metric("Hypothetical contact costs", f"{scenario['assumed_contact_cost']:,.0f}")
c.metric("Illustrative net proxy", f"{scenario['illustrative_net_proxy']:,.0f}")
st.warning("This is a retrospective observational scenario. It does not estimate incremental lift, actual profitability, or how many customers would convert because of a call. The dataset has no no-call control group.")

st.subheader("Comparing models")
comparison = []
for name, values in manifest["models"].items():
    v = values["test"]
    comparison.append({"model": name, "ROC-AUC": v["roc_auc"], "AP": v["average_precision"], "Brier": v["brier_score"], "Top 20% lift": v["top_k"]["lift_at_k"]})
st.dataframe(pd.DataFrame(comparison), hide_index=True, use_container_width=True)

feat_file = result_dir / "feature_importance.csv"
if feat_file.exists():
    st.subheader("Feature ranking (predictive association, not causal effect)")
    feature_df = pd.read_csv(feat_file).head(18)
    st.bar_chart(feature_df.set_index("feature")["importance_magnitude"])

st.divider()
st.caption("Data credit: Moro, Rita & Cortez (2014), UCI Bank Marketing, CC BY 4.0 · This repository's code: MIT.")
st.caption("Research development: Jinlei Cao. Review and reproduce all analysis before using in an academic application.")
