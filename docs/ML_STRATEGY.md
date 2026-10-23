# Machine-learning strategy

## 1. Problem framing

Survey income is often missing (about 15% in the first wave of UKHLS). Dropping those people biases results
because the missing values are **not random** (the thesis found missingness depends on age, sex,
qualification and job status). The goal is *valid statistical inference*, not just a good point prediction:
unbiased estimates, confidence intervals that cover the truth about 95% of the time, and honest uncertainty.

## 2. Success metrics

| Level | Metric | Target |
| --- | --- | --- |
| Statistical | 95% coverage of pooled intervals (simulation) | >= 0.90 for every estimand |
| Statistical | raw bias of mean wage and key regression coefficients | within 1-2% of the complete-data value |
| Predictive | RMSE of imputed vs true values (simulation) | better than mean imputation |
| Operational | endpoint p95 latency, error rate | < 500 ms warm, < 0.5% errors |
| Business | analyst time to a defensible estimate | one command or one API call |

## 3. Data strategy

* Real data stays in the licensed environment (S3 with restricted IAM); the repo ships a synthetic generator.
* Validation gate (`data.validate`): schema, ranges, unseen categories, missing rate.
* Simulation studies cover **MCAR, MAR and MNAR**; MNAR cannot be fixed by any method without extra
  assumptions, so it is reported as a sensitivity analysis, never hidden.

## 4. Modelling strategy

* Baselines first (`mean`, `sample`, complete-case `lm`) so improvements are measurable.
* Candidate set = the 11 `mice` methods from the thesis, ported to Python.
* Model on log wage (right-skewed, positive).
* Choose by **coverage first, then RMSE** (a narrow interval that misses the truth is worse than a wide one).
* Possible next steps: gradient-boosted PMM, Bayesian hierarchical models, MNAR pattern-mixture sensitivity.

## 5. MLOps maturity roadmap

| Stage | What exists | Next |
| --- | --- | --- |
| 0 Notebook / R Markdown | original thesis | - |
| 1 Tested package + CLI | done: 56 tests | add property-based tests |
| 2 Reproducible pipeline | SageMaker Pipeline written | run on AWS, store execution history |
| 3 Governed release | registry with manual approval | model cards, signed artefacts |
| 4 Monitored service | Model Monitor config documented | data-quality baseline + alarms |
| 5 Continuous improvement | - | scheduled re-simulation when new waves arrive |

## 6. Where foundation models (Bedrock) help, and where they must not

* **Helps:** explain simulation results, draft the methods paragraph, generate a data-quality summary from
  aggregate statistics.
* **Must not:** produce the imputed values, see row-level data, or make the approval decision.
* Controls: aggregate-only input check, low temperature, optional Bedrock Guardrail, outputs labelled as
  generated, human review.

## 7. Risks

| Risk | Mitigation |
| --- | --- |
| MNAR missingness | sensitivity analysis; document the assumption |
| Re-identification from donor models | default to parametric model; restrict artefact access |
| Silent drift in new survey waves | Model Monitor + re-run simulation per wave |
| Cost overrun | serverless endpoint, budget alert, delete scripts |
| Over-trusting generated text | guardrail + human sign-off |
