import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_script(name: str):
    """Import a script from ``mlops/`` by path (it is not a package)."""
    spec = importlib.util.spec_from_file_location(f"mlops_{name}", ROOT / "mlops" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def synthetic():
    from income_imputation.data import make_synthetic

    return make_synthetic(n=2500, seed=7)


@pytest.fixture(scope="session")
def fitted(synthetic):
    from income_imputation.model import ImputationModel

    return ImputationModel(method="norm", m=5, seed=1).fit(synthetic)
