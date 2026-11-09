"""Point-in-time feature policy and train-only transformations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FORBIDDEN_PREDICTORS = {"y", "duration"}


def prepare_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Make known-at-contact features; intentionally remove post-call duration."""
    x = frame.drop(columns=list(FORBIDDEN_PREDICTORS & set(frame.columns))).copy()
    if "duration" in x.columns or "y" in x.columns:
        raise AssertionError("Target or post-contact duration leaked into predictors")
    if "pdays" in x.columns:
        days = pd.to_numeric(x.pop("pdays"), errors="coerce")
        x["previous_contact_known"] = (days != 999).astype(int)
        x["days_since_previous"] = days.mask(days == 999, np.nan)
    y = frame["y"].eq("yes").astype(int)
    return x, y


def make_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numbers = list(x.select_dtypes(include=["number", "bool"]).columns)
    categories = [c for c in x.columns if c not in numbers]
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numbers),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categories),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def chronological_indices(n: int, train_frac: float = .60, valid_frac: float = .20):
    """Use original UCI row order: first train, then validation, then test."""
    if n < 10 or not (0 < train_frac < 1) or not (0 < valid_frac < 1):
        raise ValueError("Invalid split size or fractions")
    a = int(n * train_frac)
    b = int(n * (train_frac + valid_frac))
    if not (0 < a < b < n):
        raise ValueError("Split resulted in an empty partition")
    return np.arange(0, a), np.arange(a, b), np.arange(b, n)
