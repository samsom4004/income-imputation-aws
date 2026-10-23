# From the thesis to this repository

Original: *Multiple Imputation for income in UKHLS study*, University of Essex, MA831 third-year project
(2019-20), R Markdown with the `mice` package.

| Thesis | Here |
| --- | --- |
| Wage extract with 25,654 people, 3,821 missing wages (about 15%) | `data.py` schema + synthetic generator with the same shape |
| Logistic regression of the missingness indicator (depends on age, sex, qualification, job) | `missingness.py` MAR mechanism built from those findings; MCAR and MNAR added |
| 11 `mice` methods, `m = 5` | `imputers.py`: same 11 methods in Python |
| Rubin's rules | `pooling.py` |
| Design 2 simulation (missing-data mechanism); raw bias, coverage, average width, RMSE | `evaluate.py` |
| Real-data imputation with `cart` | `ImputationModel(method="cart")`, but the default is `norm` (see below) |

## What changed and why

* **Python, packaged and tested**: reproducible, importable, and runnable on SageMaker.
* **Log-wage modelling** by default: wages are skewed; imputations stay positive.
* **Default `norm` instead of `cart`** for serving: no individual data inside the artefact.
* **Synthetic data**: the licensed extract cannot be published.
* **Deployment**: API server, container, SageMaker pipeline, registry gate, monitoring, cost controls.
* **Bedrock**: explanatory reports from aggregate metrics only.

## Honest limitations

* The synthetic generator is a plausible stand-in, not the real distribution. Re-run the study on the real
  extract inside your licensed environment before trusting any ranking of methods.
* The thesis ranked methods on the real data; the numbers here come from synthetic data and will differ.
* MNAR cannot be corrected by any imputation method without extra assumptions.
