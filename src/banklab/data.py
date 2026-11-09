"""Download and validate the official UCI Bank Marketing dataset."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/222/bank%2Bmarketing.zip"
UCI_DOI = "https://doi.org/10.24432/C5K306"
CSV_NAME = "bank-additional-full.csv"


class DatasetError(ValueError):
    """The supplied data does not match the expected structure."""


def find_csv_in_zip(blob: bytes, depth: int = 0) -> bytes:
    """Find the named CSV in the official archive (also handles nested ZIPs)."""
    if depth > 3:
        raise DatasetError("ZIP nesting too deep")
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for member in archive.namelist():
            if Path(member).name.lower() == CSV_NAME.lower():
                return archive.read(member)
        for member in archive.namelist():
            if member.lower().endswith(".zip"):
                try:
                    return find_csv_in_zip(archive.read(member), depth + 1)
                except (DatasetError, zipfile.BadZipFile):
                    continue
    raise DatasetError(f"Could not locate {CSV_NAME} in downloaded archive")


def download_uci(destination: str | Path, *, replace: bool = False) -> Path:
    """Retrieve the UCI archive and extract the original semicolon-separated CSV."""
    destination = Path(destination)
    if destination.exists() and not replace:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(UCI_URL, headers={"User-Agent": "BankMarketingDecisionLab/0.1"})
    with urlopen(request, timeout=60) as response:
        archive_bytes = response.read()
    csv_bytes = find_csv_in_zip(archive_bytes)
    frame = pd.read_csv(io.BytesIO(csv_bytes), sep=";")
    validate_data(frame)
    if len(frame) != 41_188:
        raise DatasetError(f"Expected 41,188 records from {CSV_NAME}; got {len(frame):,}")
    destination.write_bytes(csv_bytes)
    return destination


def validate_data(frame: pd.DataFrame) -> None:
    required = {"y", "duration", "age", "campaign", "pdays", "previous", "poutcome"}
    missing = required - set(frame.columns)
    if missing:
        raise DatasetError(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        raise DatasetError("Dataset is empty")
    bad = set(frame["y"].astype(str).unique()) - {"yes", "no"}
    if bad:
        raise DatasetError(f"Target must contain yes/no values, found: {bad}")


def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}. Run: banklab download")
    frame = pd.read_csv(path, sep=";")
    validate_data(frame)
    return frame


def make_synthetic_demo(n: int = 1_600, seed: int = 42) -> pd.DataFrame:
    """Generate artificial data with UCI-shaped columns; NEVER treat as evidence."""
    if n < 200:
        raise ValueError("Use at least 200 synthetic rows for three-way evaluation")
    rng = np.random.default_rng(seed)
    months = np.array(["may", "jun", "jul", "aug", "sep", "oct", "nov"])
    jobs = np.array(["admin.", "blue-collar", "management", "technician", "services", "student"])
    prev = rng.choice([0, 1, 2, 3], n, p=[.76, .17, .05, .02])
    pout = np.where(prev == 0, "nonexistent", rng.choice(["success", "failure"], n, p=[.25, .75]))
    age = np.clip(rng.normal(40, 11, n).astype(int), 18, 80)
    phase = np.linspace(-1, 1, n)
    frame = pd.DataFrame({
        "age": age,
        "job": rng.choice(jobs, n),
        "marital": rng.choice(["single", "married", "divorced"], n),
        "education": rng.choice(["high.school", "university.degree", "basic.9y", "unknown"], n),
        "default": rng.choice(["no", "unknown"], n, p=[.8, .2]),
        "housing": rng.choice(["yes", "no", "unknown"], n),
        "loan": rng.choice(["no", "yes", "unknown"], n),
        "contact": rng.choice(["cellular", "telephone"], n),
        "month": months[np.arange(n) % len(months)],
        "day_of_week": rng.choice(["mon", "tue", "wed", "thu", "fri"], n),
        "duration": np.zeros(n, dtype=int),
        "campaign": rng.integers(1, 8, n),
        "pdays": np.where(prev == 0, 999, rng.integers(1, 25, n)),
        "previous": prev,
        "poutcome": pout,
        "emp.var.rate": -0.3 + phase + rng.normal(0, .2, n),
        "cons.price.idx": 93.5 + rng.normal(0, .4, n),
        "cons.conf.idx": -41 + rng.normal(0, 4, n),
        "euribor3m": 2.3 + phase + rng.normal(0, .25, n),
        "nr.employed": 5150 + 70 * phase + rng.normal(0, 15, n),
    })
    z = (
        -2.30 + 1.05 * (frame["poutcome"].eq("success").to_numpy())
        + 0.45 * (frame["job"].eq("student").to_numpy())
        - 0.11 * (frame["campaign"].to_numpy() - 1)
        + 0.30 * (frame["contact"].eq("cellular").to_numpy())
        - 0.25 * phase
    )
    prob = 1.0 / (1 + np.exp(-z))
    subscribed = rng.random(n) < prob
    frame["y"] = np.where(subscribed, "yes", "no")
    # Post-outcome duration is intentionally correlated with outcome.
    frame["duration"] = np.maximum(1, (rng.gamma(2, 60, n) + 80 * subscribed).astype(int))
    validate_data(frame)
    return frame
