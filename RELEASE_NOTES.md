# Release v0.1.0 — First GitHub-ready project package

Repository target: https://github.com/jinlei-cao/bank-marketing-decision-lab

**Delivered:**
- Official UCI data downloader (requires internet when run)
- Pre-call feature policy and 60/20/20 chronological validation
- Logistic regression and random forest comparative research pipeline
- Average precision, ROC-AUC, calibration and top-capacity lift metrics
- Scenario dashboard (optional Streamlit dependency)
- Synthetic offline results in `reports/demo/` (EXAMPLE ONLY)
- Validated Jupyter notebook
- Nine automated pytest checks and GitHub Actions workflow
- MIT source license, UCI CC BY 4.0 attribution and disclosure of AI-assisted initial scaffold

**Verification in the packaging environment:**
- Nine tests passed.
- `banklab demo` ran successfully and generated all expected artifacts.
- Notebook executed through nbclient (four code cells).
- Streamlit runtime could not be tested here (optional dependency unavailable offline).
- Official UCI CSV could not be downloaded from this environment (network DNS blocked); therefore NO empirical model performance claim is shipped.

**Next step:** Download and train the real UCI dataset on an internet-enabled machine, inspect results, and then publish verified findings in GitHub README.
