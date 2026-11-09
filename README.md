# Bank Marketing Decision Lab

**Predicting term-deposit subscriptions and evaluating data-driven contact allocation**  
A reproducible, open-source quantitative research portfolio for **Jinlei Cao**.

> **Status: v0.1 research toolkit.** Synthetic demo works offline. Real empirical results are generated only after downloading the official UCI data and executing the analysis. No model performance claim is made before that run.

## Research question

Can historical customer and campaign information, available **before a call**, predict term-deposit subscriptions well enough to prioritise scarce marketing contacts? What happens to evaluation when the model faces later campaigns rather than a random historical sample?

This project combines *econometrics and business analytics* with *bank marketing strategy* and is designed for transparent review by postgraduate admissions committees and other researchers.

## Why this project is different

- **Point-in-time honesty:** excludes `duration`, a post-call field that would overstate realistic predictive performance.
- **Chronological holdout:** 60% train → 20% validation → 20% test, following the source file's known ordering.
- **Interpretable baseline:** logistic regression versus random forest, champion chosen on **validation AP**.
- **Decision metrics:** precision@capacity, recall@capacity and lift against random selection.
- **Interactive dashboard:** adjust contact capacity and *assumed* economic inputs, then inspect retrospective selection outcomes.
- **Research integrity:** synthetic examples are explicitly labelled; historical subscriptions are not causal uplift.

## Quick start (Python 3.10+)

```bash
# Unzip the package; run commands from its top-level folder
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -e '.[app,dev]'

# Start with an OFFLINE SYNTHETIC demonstration (NOT research data)
banklab demo
streamlit run app/streamlit_app.py
```

The dashboard initially points to `reports/demo` and will show a red synthetic-data warning. This is useful for interface testing; never cite its metrics as real research findings.

### Run the real UCI experiment

```bash
banklab download
banklab train --data data/raw/bank-additional-full.csv --output reports/latest
streamlit run app/streamlit_app.py
```

Enter `reports/latest` in the dashboard sidebar. The command saves:

- `metrics.json` — model comparison and test metrics
- `holdout_predictions.csv` — pseudonymous row indices, observed outcomes and prediction scores
- `feature_importance.csv` — magnitudes of model associations
- `precision_recall.png`, `targeting_lift.png` — reproducible figures
- `RESULTS.md` — generated summary of observed results
- `champion_model.joblib` — trained pipeline for local study

**Connectivity:** `banklab download` needs internet access once; `banklab demo` and unit tests work offline.

## Repository structure

```text
bank-marketing-decision-lab/
├── README.md                     # Research overview and instructions
├── pyproject.toml                 # Python dependencies + CLI
├── CITATION.cff                   # Software citation
├── LICENSE                        # MIT software license
├── src/banklab/
│   ├── data.py                    # UCI source, validation, demo generator
│   ├── features.py                # No leakage, feature engineering, split
│   ├── metrics.py                 # Validation and capacity-based economics
│   ├── experiment.py              # Training, selection, figures and outputs
│   └── cli.py                     # download / demo / train
├── app/streamlit_app.py           # Interactive dashboard
├── notebooks/01_exploration.ipynb # Optional research exploration
├── tests/                         # Automated quality checks
├── docs/                          # Research plan, methods and limitations
├── data/README.md                 # Data provenance; raw CSV excluded
└── .github/workflows/tests.yml    # GitHub Actions
```

## How to interpret outputs

The 41,188-record, 20-input `bank-additional-full.csv` is from Portuguese bank campaigns in **2008–2010**. The full UCI archive also contains other variants; this project uses only the named file. The UCI source identifies `duration` as known only after a call. The model therefore does not use it. The chosen split also preserves the original chronological ordering (though it does not provide date-perfect boundaries).

**Prediction ≠ causation.** Because all observations came from customers already contacted, neither precision@K nor its financial scenario measures the effect of calling rather than not calling. The result is a ranking study, not a validated marketing policy or investment return.

Read [METHODOLOGY.md](docs/METHODOLOGY.md) for methods, modelling risks, privacy and limitations. This project is for research/education, not real credit or consumer targeting decisions.

## Data source and academic citation

Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306

Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems, 62*, 22–31. https://doi.org/10.1016/j.dss.2014.03.001
