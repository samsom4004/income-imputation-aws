import json

from income_imputation import cli


def test_cli_train_impute_and_explain(tmp_path, capsys, synthetic):
    model = tmp_path / "model.joblib"
    cli.main(["train", "--n", "1200", "--out", str(model), "--method", "norm"])
    inp = tmp_path / "in.csv"
    synthetic.drop(columns=["wage_true"]).head(200).to_csv(inp, index=False)
    out = tmp_path / "out.csv"
    cli.main(["impute", "--model", str(model), "--input", str(inp), "--output", str(out)])
    assert out.exists() and "rows imputed" in capsys.readouterr().out
    m = tmp_path / "metrics.json"
    cli.main(["simulate", "--n", "1200", "--n-sim", "2", "--methods", "mean,norm", "--out", str(m)])
    assert "mean_wage" in json.loads(m.read_text())["best_method_by_estimand"]
    cli.main(["explain", "--metrics", str(m), "--dry-run"])
    assert "Imputation study" in capsys.readouterr().out
