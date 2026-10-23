import json
import tarfile

from conftest import load_script


def test_train_script_writes_model_and_metrics(tmp_path):
    train = load_script("train")
    train.main(["--synthetic", "--synthetic-n", "1500", "--n-sim", "3", "--model-dir", str(tmp_path / "model"),
                "--output-dir", str(tmp_path / "out")])
    assert (tmp_path / "model" / "model.joblib").exists()
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert metrics["selected_method"] == "norm" and 0 <= metrics["min_coverage"] <= 1
    assert metrics["contains_individual_data"] is False


def test_inference_handlers_roundtrip(tmp_path, fitted):
    inf = load_script("inference")
    fitted.save(tmp_path / "model.joblib")
    model = inf.model_fn(str(tmp_path))
    rec = {"age": 50, "sex": "Male", "health": "No", "hiqual": "Degree", "ethnic": "White",
           "marriage": "Have partner", "jb": "FT employed"}
    data = inf.input_fn(json.dumps({"records": [rec], "m": 4}), "application/json")
    body, ctype = inf.output_fn(inf.predict_fn(data, model))
    assert ctype == "application/json" and json.loads(body)["predictions"][0]["wage_imputed"] > 0


def test_inference_rejects_bad_input():
    import pytest

    inf = load_script("inference")
    with pytest.raises(ValueError):
        inf.input_fn("{}", "text/csv")
    with pytest.raises(ValueError):
        inf.input_fn(json.dumps({"records": []}))


def test_prepare_then_train_then_evaluate_like_the_pipeline(tmp_path):
    prep, train, ev = load_script("prepare"), load_script("train"), load_script("evaluate_step")
    prep.main(base=str(tmp_path))
    assert (tmp_path / "output" / "train" / "train.csv").exists() and (tmp_path / "output" / "test" / "test.csv").exists()
    train.main(["--train", str(tmp_path / "output" / "train"), "--n-sim", "3", "--model-dir", str(tmp_path / "m"),
                "--output-dir", str(tmp_path / "o")])
    (tmp_path / "model").mkdir()
    with tarfile.open(tmp_path / "model" / "model.tar.gz", "w:gz") as tar:
        tar.add(tmp_path / "m" / "model.joblib", arcname="model.joblib")
    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "test.csv").write_bytes((tmp_path / "output" / "test" / "test.csv").read_bytes())
    ev.main(base=str(tmp_path))
    report = json.loads((tmp_path / "evaluation" / "evaluation.json").read_text())
    assert "value" in report["metrics"]["min_coverage"]
