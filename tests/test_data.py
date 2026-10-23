import pandas as pd

from income_imputation.data import CATEGORICALS, COVARIATES, design_matrix, load_csv, make_synthetic, validate


def test_synthetic_shape_and_missing_rate(synthetic):
    assert len(synthetic) == 2500
    assert set(COVARIATES + ["wage", "wage_true", "wage_missing"]).issubset(synthetic.columns)
    assert abs(synthetic["wage"].isna().mean() - 0.15) < 0.03
    assert (synthetic["wage_true"] > 0).all()
    assert validate(synthetic) == []


def test_synthetic_is_reproducible():
    a, b = make_synthetic(n=300, seed=3), make_synthetic(n=300, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_missingness_is_mar_not_mcar(synthetic):
    # people with a degree should be missing more often than people with no qualification
    rate = synthetic.groupby("hiqual")["wage_missing"].mean()
    assert rate["Degree"] > rate["No qual"]


def test_design_matrix_has_stable_columns_and_handles_unknown_levels(synthetic):
    full = design_matrix(synthetic)
    one = synthetic.iloc[:1].copy()
    one["hiqual"] = "Never seen before"
    row = design_matrix(one)
    assert list(row.columns) == list(full.columns)
    assert row["hiqual=A-level"].iloc[0] == 0.0  # unknown level falls back to the reference level
    assert full.shape[1] == 1 + sum(len(v) - 1 for v in CATEGORICALS.values())


def test_validate_flags_problems(synthetic):
    bad = synthetic.copy()
    bad.loc[0, "age"] = 500
    bad.loc[1, "sex"] = "Robot"
    problems = validate(bad)
    assert any("age" in p for p in problems) and any("sex" in p for p in problems)
    assert validate(synthetic.drop(columns=["wage"])) == ["missing column: wage"]


def test_load_csv_handles_missing_marker_and_job_columns(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text(
        "id,sex,age,jbstat,health,hiqual,jbft,ethnic,marriage,wage,wage_missing\n"
        "1,Male,22,Employed,No,GCSE,PT,Asian,Single,450,0\n"
        "2,Female,40,Retired,No,Degree,FT,White,Have partner,missing,1\n"
    )
    df = load_csv(str(path))
    assert df["wage"].isna().tolist() == [False, True]
    assert df["jb"].tolist() == ["PT employed", "Retired"]
