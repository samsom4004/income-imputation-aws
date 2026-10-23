import numpy as np

from income_imputation.pooling import rubin


def test_single_imputation_is_a_plain_normal_interval():
    r = rubin([10.0], [4.0])
    assert r["qbar"] == 10.0 and abs((r["hi"] - r["lo"]) / 2 - 1.96 * 2) < 0.01


def test_rubins_rules_match_hand_calculation():
    q, u = np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.5, 0.5])
    r = rubin(q, u)
    assert r["qbar"] == 2.0 and r["ubar"] == 0.5 and abs(r["b"] - 1.0) < 1e-12
    assert abs(r["t"] - (0.5 + (1 + 1 / 3) * 1.0)) < 1e-12  # T = U + (1 + 1/m) B


def test_more_between_imputation_variance_widens_the_interval():
    tight = rubin([5.0, 5.01, 4.99], [1, 1, 1])
    loose = rubin([3.0, 5.0, 7.0], [1, 1, 1])
    assert loose["hi"] - loose["lo"] > tight["hi"] - tight["lo"]
