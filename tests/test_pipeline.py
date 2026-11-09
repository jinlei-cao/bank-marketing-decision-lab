import json

import pandas as pd

from banklab.data import make_synthetic_demo
from banklab.experiment import run_experiment


def test_end_to_end_synthetic_pipeline(tmp_path):
    df = make_synthetic_demo(520, seed=42)
    manifest = run_experiment(df, tmp_path, source="SYNTHETIC DEMO — NOT EMPIRICAL EVIDENCE", fast=True)
    assert manifest["synthetic_demo"] is True
    assert "duration" in manifest["excluded_fields"]
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "precision_recall.png").exists()
    assert (tmp_path / "champion_model.joblib").exists()
    scores = pd.read_csv(tmp_path / "holdout_predictions.csv")
    assert len(scores) == 104
    assert scores["p_champion"].between(0, 1).all()
    assert json.loads((tmp_path / "metrics.json").read_text())["champion"] in manifest["models"]
