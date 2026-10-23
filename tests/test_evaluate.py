from income_imputation.evaluate import run_simulation, summarise


def test_proper_methods_beat_mean_imputation_under_mar(synthetic):
    table = run_simulation(synthetic, ["mean", "norm"], mechanism="mar", rate=0.2, n_sim=6, m=5, seed=0)
    get = lambda m, e, c: float(table[(table["method"] == m) & (table["estimand"] == e)][c].iloc[0])  # noqa: E731
    assert abs(get("norm", "mean_wage", "raw_bias")) < abs(get("mean", "mean_wage", "raw_bias"))
    assert get("norm", "coef_no_qual", "coverage") >= get("mean", "coef_no_qual", "coverage")
    assert get("norm", "coef_no_qual", "rmse") < get("mean", "coef_no_qual", "rmse")


def test_lm_runs_and_summary_is_json_friendly(synthetic):
    import json

    table = run_simulation(synthetic, ["lm", "pmm"], n_sim=3, m=3, seed=1)
    s = summarise(table, "mar", 0.15, 3, 3)
    json.dumps(s)
    assert set(s["best_method_by_estimand"]) == {"mean_wage", "coef_age", "coef_no_qual"}
