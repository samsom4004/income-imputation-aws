import numpy as np
import pytest

from income_imputation.imputers import METHODS, impute_values


@pytest.fixture()
def arrays():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4))
    y = 3 + X @ np.array([1.0, -0.5, 0.2, 0.0]) + rng.normal(0, 0.5, 400)
    return X[:300], y[:300], X[300:], y[300:]


@pytest.mark.parametrize("method", [m for m in METHODS if m != "lm"])
def test_every_method_returns_finite_values_of_right_length(method, arrays):
    Xo, yo, Xm, _ = arrays
    out = impute_values(method, Xo, yo, Xm, np.random.default_rng(1))
    assert out.shape == (len(Xm),) and np.isfinite(out).all()


def test_lm_means_complete_case(arrays):
    Xo, yo, Xm, _ = arrays
    assert impute_values("lm", Xo, yo, Xm, np.random.default_rng(0)) is None


def test_unknown_method_raises(arrays):
    Xo, yo, Xm, _ = arrays
    with pytest.raises(ValueError):
        impute_values("magic", Xo, yo, Xm, np.random.default_rng(0))


def test_no_missing_rows_gives_empty_result(arrays):
    Xo, yo, _, _ = arrays
    assert len(impute_values("pmm", Xo, yo, Xo[:0], np.random.default_rng(0))) == 0


@pytest.mark.parametrize("method", ["pmm", "midastouch", "sample", "cart", "rf"])
def test_donor_methods_only_return_observed_values(method, arrays):
    Xo, yo, Xm, _ = arrays
    out = impute_values(method, Xo, yo, Xm, np.random.default_rng(2))
    assert np.isin(out, yo).all()


def test_same_seed_same_result_different_seed_differs(arrays):
    Xo, yo, Xm, _ = arrays
    a = impute_values("norm", Xo, yo, Xm, np.random.default_rng(5))
    b = impute_values("norm", Xo, yo, Xm, np.random.default_rng(5))
    c = impute_values("norm", Xo, yo, Xm, np.random.default_rng(6))
    assert np.allclose(a, b) and not np.allclose(a, c)


def test_regression_methods_beat_mean_on_error(arrays):
    Xo, yo, Xm, ytrue = arrays
    rmse = lambda v: float(np.sqrt(np.mean((v - ytrue) ** 2)))  # noqa: E731
    mean = rmse(impute_values("mean", Xo, yo, Xm, np.random.default_rng(0)))
    pred = rmse(impute_values("norm.predict", Xo, yo, Xm, np.random.default_rng(0)))
    assert pred < mean
