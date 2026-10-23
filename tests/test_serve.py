import pytest
from fastapi.testclient import TestClient

REC = {"age": 33, "sex": "Female", "health": "No", "hiqual": "GCSE", "ethnic": "Asian",
       "marriage": "Single", "jb": "PT employed"}


@pytest.fixture()
def client(fitted, tmp_path, monkeypatch):
    fitted.save(tmp_path / "model.joblib")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    from income_imputation import serve

    serve._state["model"] = None
    return TestClient(serve.app)


def test_ping_and_invocations(client):
    assert client.get("/ping").json() == {"status": "ok"}
    r = client.post("/invocations", json={"records": [REC], "m": 3})
    assert r.status_code == 200 and r.json()["predictions"][0]["wage_imputed"] > 0


def test_bare_list_is_accepted(client):
    assert client.post("/invocations", json=[REC]).status_code == 200


@pytest.mark.parametrize("body,code", [({"records": []}, 400), ({"records": [{"age": 1}]}, 422),
                                       ({"records": [REC], "m": 500}, 400), ("nope", 400)])
def test_bad_requests_get_clear_errors(client, body, code):
    r = client.post("/invocations", json=body) if not isinstance(body, str) else client.post("/invocations", content=body)
    assert r.status_code == code


def test_too_many_records(client):
    assert client.post("/invocations", json={"records": [REC] * 1001}).status_code == 413


def test_ping_without_model_is_503(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    from income_imputation import serve

    serve._state["model"] = None
    assert TestClient(serve.app).get("/ping").status_code == 503
