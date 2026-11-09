# Data provenance

**Source:** Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306

**Dataset license:** Creative Commons Attribution 4.0 (CC BY 4.0) as specified on UCI.

Use `banklab download` to retrieve the original `bank-additional-full.csv` (41,188 records, 20 predictors + binary outcome). This repository **does not contain the real customer-contact dataset**. Do not substitute the older `bank-full.csv`: it has a different feature schema.

One row represents a campaign contact rather than a confirmed unique customer. Repeated-client identifiers are not supplied. The `duration` field is only known after a call and is excluded from deployable pre-contact modelling. `pdays == 999` indicates no previous contact; it is not 999 elapsed days.

The historical data cover a Portuguese bank in 2008–2010, so conclusions should not be extrapolated to current China or Hong Kong banking without new evidence.

For privacy and reproducibility, generated model artifacts, predictions and raw data are gitignored.
