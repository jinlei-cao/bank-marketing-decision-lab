# Methodology and interpretation guide

## Decision context

The decision being supported is **which previously contacted or available customers should receive a campaign call, at a fixed contact capacity**. The observed outcome is historical term-deposit subscription among contacted records; it is **not causal uplift** from a phone call. A control group or randomized experiment would be needed to estimate incremental effect.

## Feature eligibility

- `duration` is excluded because it occurs after the attempted contact.
- `y` is strictly the label, never a predictor.
- `pdays == 999` is expanded into `previous_contact_known` and a missing `days_since_previous` value.
- `unknown` categorical labels are preserved as categories.
- Contact type and month/day are presumed known at the campaign scheduling decision. If scoring earlier than scheduling, remove those fields too and retrain.
- Previous contact counts, social indicators and macro variables are only valid if recorded before the targeting decision.

## Validation strategy

The dataset file is ordered by time, so 60% of records train the models, the next 20% select a champion using **validation average precision**, and the final 20% are an untouched holdout for model reporting. Random train/test splitting is avoided as the main method. The original data do not provide exact calendar dates for robust month-level grouping; row order is an imperfect proxy.

Train-only pipelines fit imputation, standardisation and one-hot encodings. Compared models: unweighted logistic regression and a regularised random forest. Predefined hyperparameters avoid test-set optimisation. Report precision–recall AUC (average precision), ROC-AUC, Brier score, log loss and top-capacity precision, recall and lift.

## Commercial decision layer

Select the top K% of scored holdout observations and compare historically observed subscriptions with expected random selection. The dashboard's subscription values and call costs are **assumed** inputs. The scenario is not a causal return-on-investment estimate: the bank may have obtained many subscriptions without contacting these customers.

## Risks

- Historical conditions and concept drift; rates and customer preferences have changed since 2010.
- Repeated clients, unknown sampling mechanism and selection bias from the bank's previous contact policy.
- Uncertain availability of some contact features at the true decision point.
- Potential fairness and consumer-protection implications when using age, marital status and other personal attributes. Any deployment requires legal review, appropriate consent and privacy protections; the project is for educational research only.
- Model weights, coefficients and predictive feature importance do not establish causation.
- Temporal split may experience major prevalence shift; precision at capacity and calibration must be evaluated on the target period.

## Extending to production-style research

- Add known point-in-time feature snapshots and customer IDs to address repeated clients.
- Add confidence intervals by time-aware bootstrap, no-contact randomized holdout and uplift estimation.
- Add groupwise performance audits where lawful and meaningful.
- Compare calibrated gradient boosting and financial utility under defensible value and cost estimates.
