"""Evaluation and decision metrics with explicit economic assumptions."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, roc_auc_score,
)


def ranking_at_capacity(y_true, scores, fraction: float) -> dict:
    """Retrospective targeting metrics, NOT estimates of incremental causal impact."""
    truth = np.asarray(y_true, dtype=int)
    probs = np.asarray(scores, dtype=float)
    if len(truth) == 0 or len(truth) != len(probs):
        raise ValueError("Nonempty arrays with identical length are required")
    if not (0 < fraction <= 1):
        raise ValueError("Capacity must be in (0,1]")
    n = len(truth)
    k = max(1, int(np.ceil(n * fraction)))
    chosen = np.argsort(-probs, kind="stable")[:k]
    hits = int(truth[chosen].sum())
    positives = int(truth.sum())
    baseline = positives / n
    precision = hits / k
    return {
        "capacity_fraction": float(fraction),
        "contacts": k,
        "observed_subscriptions": hits,
        "precision_at_k": precision,
        "recall_at_k": (hits / positives) if positives else 0.0,
        "lift_at_k": (precision / baseline) if baseline else None,
        "random_expected_subscriptions": k * baseline,
        "predicted_subscriptions": float(probs[chosen].sum()),
    }


def classification_metrics(y_true, scores, capacity: float = .20) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(scores, dtype=float)
    if np.any(~np.isfinite(p)):
        raise ValueError("Probabilities must be finite")
    p_safe = np.clip(p, 1e-12, 1 - 1e-12)
    return {
        "samples": int(len(y)),
        "positive_count": int(y.sum()),
        "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, p)) if len(set(y)) == 2 else None,
        "brier_score": float(brier_score_loss(y, p_safe)),
        "log_loss": float(log_loss(y, p_safe, labels=[0, 1])),
        "top_k": ranking_at_capacity(y, p, capacity),
    }


def illustrative_scenario(top_k: dict, value_per_subscription: float, cost_per_contact: float) -> dict:
    """Gross-value assumptions only, not experimental uplift or realised profit."""
    if value_per_subscription < 0 or cost_per_contact < 0:
        raise ValueError("Assumptions must be nonnegative")
    return {
        "assumed_gross_value": top_k["observed_subscriptions"] * value_per_subscription,
        "assumed_contact_cost": top_k["contacts"] * cost_per_contact,
        "illustrative_net_proxy": (
            top_k["observed_subscriptions"] * value_per_subscription
            - top_k["contacts"] * cost_per_contact
        ),
    }
