"""Train, select, evaluate and export a reproducible research experiment."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.pipeline import Pipeline

from .data import validate_data
from .features import chronological_indices, make_preprocessor, prepare_features
from .metrics import classification_metrics, ranking_at_capacity


def _make_models(x_train: pd.DataFrame, *, fast: bool = False) -> dict[str, Pipeline]:
    def pre():
        return make_preprocessor(x_train)
    return {
        "logistic_regression": Pipeline([
            ("preprocess", pre()),
            ("model", LogisticRegression(max_iter=1_200, random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("preprocess", pre()),
            ("model", RandomForestClassifier(
                n_estimators=40 if fast else 140,
                min_samples_leaf=8 if fast else 20,
                max_depth=10, random_state=42, n_jobs=-1,
            )),
        ]),
    }


def _plot_curves(test_y: pd.Series, preds: dict[str, np.ndarray], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for name, p in preds.items():
        PrecisionRecallDisplay.from_predictions(test_y, p, name=name.replace("_", " "), ax=ax)
    ax.axhline(float(test_y.mean()), ls="--", c="0.5", label="test prevalence")
    ax.set_title("Chronological holdout: precision–recall")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_targeting(test_y: pd.Series, p: np.ndarray, output: Path) -> None:
    fractions = np.linspace(.05, 1.0, 20)
    lifts = [ranking_at_capacity(test_y, p, float(f))["lift_at_k"] or 0 for f in fractions]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(fractions * 100, lifts, color="black", lw=2, label="champion")
    ax.axhline(1.0, color="0.5", ls="--", label="random baseline")
    ax.set(xlabel="Contact capacity (% of holdout)", ylabel="Lift versus random", title="Retrospective targeting lift")
    ax.grid(alpha=.15)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_experiment(frame: pd.DataFrame, output_dir: str | Path, *, source: str, fast: bool = False) -> dict:
    """Select champion by validation AP, evaluate once on final chronological test."""
    validate_data(frame)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    x, y = prepare_features(frame)
    train_idx, valid_idx, test_idx = chronological_indices(len(x))
    x_train, x_val, x_test = x.iloc[train_idx], x.iloc[valid_idx], x.iloc[test_idx]
    y_train, y_val, y_test = y.iloc[train_idx], y.iloc[valid_idx], y.iloc[test_idx]
    results = {}
    fitted = {}
    test_predictions = {}
    for name, model in _make_models(x_train, fast=fast).items():
        model.fit(x_train, y_train)
        val_p = model.predict_proba(x_val)[:, 1]
        test_p = model.predict_proba(x_test)[:, 1]
        results[name] = {
            "validation": classification_metrics(y_val, val_p),
            "test": classification_metrics(y_test, test_p),
        }
        fitted[name] = model
        test_predictions[name] = test_p
    champion = max(results, key=lambda k: results[k]["validation"]["average_precision"] or -1)
    champion_model = fitted[champion]
    predictions = pd.DataFrame({
        "source_row_index": test_idx,
        "observed_subscribed": y_test.to_numpy(),
        **{f"p_{name}": p for name, p in test_predictions.items()},
        "p_champion": test_predictions[champion],
    })
    predictions.to_csv(output / "holdout_predictions.csv", index=False)
    feature_names = champion_model.named_steps["preprocess"].get_feature_names_out()
    est = champion_model.named_steps["model"]
    importance = (np.abs(est.coef_[0]) if hasattr(est, "coef_") else est.feature_importances_)
    feats = pd.DataFrame({"feature": feature_names, "importance_magnitude": importance})
    feats.sort_values("importance_magnitude", ascending=False).to_csv(output / "feature_importance.csv", index=False)
    joblib.dump(champion_model, output / "champion_model.joblib")
    _plot_curves(y_test, test_predictions, output / "precision_recall.png")
    _plot_targeting(y_test, test_predictions[champion], output / "targeting_lift.png")
    manifest = {
        "dataset_source": source,
        "synthetic_demo": source == "SYNTHETIC DEMO — NOT EMPIRICAL EVIDENCE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_records": len(frame),
        "feature_count_pre_engineering": int(frame.shape[1] - 2),
        "excluded_fields": ["duration", "y"],
        "chronological_split": {
            "train_rows": len(train_idx), "validation_rows": len(valid_idx), "test_rows": len(test_idx),
            "train_positive_rate": float(y_train.mean()),
            "validation_positive_rate": float(y_val.mean()),
            "test_positive_rate": float(y_test.mean()),
        },
        "model_selection": "Highest validation average precision; no test-set tuning",
        "champion": champion,
        "models": results,
        "limitations": [
            "Historical contact data (2008–2010); does not establish current market behaviour.",
            "Client identifiers are unavailable; repeated-client dependence cannot be removed.",
            "Predicting subscription among contacted clients is not estimating incremental campaign uplift.",
            "Macro indicators and historical regime may drift in deployment.",
            "Costs and value inputs in the dashboard are user-supplied hypothetical assumptions.",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    test = results[champion]["test"]
    note = (
        f"# Reproducible experiment results\n\n"
        f"**Dataset:** {source}\n\n"
        f"**Champion:** {champion.replace('_', ' ')} (selected by validation AP).\n\n"
        f"**Chronological test:** {test['samples']:,} observations; "
        f"positive rate {test['positive_rate']:.2%}; "
        f"ROC-AUC {test['roc_auc']:.3f}; AP {test['average_precision']:.3f}; "
        f"Brier {test['brier_score']:.3f}.\n\n"
        f"**Top 20% capacity:** {test['top_k']['contacts']:,} contacts, "
        f"{test['top_k']['observed_subscriptions']:,} observed subscriptions, "
        f"precision {test['top_k']['precision_at_k']:.2%}, "
        f"lift {test['top_k']['lift_at_k']:.2f}x.\n\n"
        "These are observational, retrospective rankings. They are not evidence of "
        "incremental conversions caused by calling.\n\n"
    )
    if manifest["synthetic_demo"]:
        note = "> **SYNTHETIC DATA: demonstration only. Not a finding about real customers.**\n\n" + note
    (output / "RESULTS.md").write_text(note, encoding="utf-8")
    return manifest
