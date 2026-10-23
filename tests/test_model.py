import numpy as np
import pytest

from income_imputation.model import ImputationModel


def test_transform_fills_missing_and_keeps_observed(fitted, synthetic):
    out = fitted.transform(synthetic)
    miss = synthetic["wage"].isna()
    assert out["wage_imputed"].notna().all()
    assert (out.loc[~miss, "wage_imputed"] == synthetic.loc[~miss, "wage"]).all()
    assert out["was_imputed"].sum() == miss.sum()
    assert (out.loc[miss, "wage_p05"] <= out.loc[miss, "wage_p95"]).all()
    assert (out["wage_imputed"] > 0).all()


def test_imputed_wages_are_plausible_against_truth(fitted, synthetic):
    miss = synthetic["wage"].isna()
    est = fitted.transform(synthetic).loc[miss, "wage_imputed"]
    truth = synthetic.loc[miss, "wage_true"]
    assert abs(np.log(est).mean() - np.log(truth).mean()) < 0.15
    assert np.corrcoef(np.log(est), np.log(truth))[0, 1] > 0.5


def test_save_load_roundtrip(fitted, synthetic, tmp_path):
    path = fitted.save(tmp_path / "m" / "model.joblib")
    again = ImputationModel.load(path)
    a = fitted.draws(synthetic.head(50), seed=3)
    b = again.draws(synthetic.head(50), seed=3)
    assert np.allclose(a, b, equal_nan=True)


def test_predict_records_api(fitted):
    rec = {"age": 40, "sex": "Male", "health": "No", "hiqual": "Degree", "ethnic": "White",
           "marriage": "Single", "jb": "FT employed"}
    out = fitted.predict_records([rec, rec], m=7, seed=1)
    assert len(out) == 2 and set(out[0]) == {"wage_imputed", "wage_p05", "wage_p95", "was_imputed"}
    assert out[0]["was_imputed"] == 1 and out[0]["wage_p05"] <= out[0]["wage_imputed"] <= out[0]["wage_p95"]


def test_predict_records_rejects_missing_fields(fitted):
    with pytest.raises(ValueError, match="missing required fields"):
        fitted.predict_records([{"age": 40}])


def test_fit_needs_enough_data(synthetic):
    with pytest.raises(ValueError):
        ImputationModel().fit(synthetic.dropna(subset=["wage"]).head(10))


def test_unfitted_model_raises():
    with pytest.raises(RuntimeError):
        ImputationModel().draws(__import__("pandas").DataFrame())


def test_lm_cannot_impute(synthetic):
    with pytest.raises(ValueError):
        ImputationModel(method="lm").fit(synthetic).draws(synthetic)


def test_privacy_flag_and_donor_cap(synthetic):
    norm = ImputationModel(method="norm").fit(synthetic)
    pmm = ImputationModel(method="pmm", max_donors=200).fit(synthetic)
    assert not norm.contains_individual_data and pmm.contains_individual_data
    assert len(pmm.y_obs_) == 200
