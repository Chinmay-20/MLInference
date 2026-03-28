from __future__ import annotations

from fastapi.testclient import TestClient

from service.app import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "ML HPC Inference Service is running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "cpp_binary_exists" in body


def test_predict():
    response = client.post(
        "/predict",
        json={"input": [1,2,3,4,5,6,7,8,9,10]}
    )
    assert response.status_code == 200
    body = response.json()
    assert "predictions" in body
    assert body["batch_size"] == 1
    assert body["output_dim"] == 1
    assert len(body["predictions"]) == 1


def test_predict_batch():
    response = client.post(
        "/predict_batch",
        json={
            "inputs": [
                [1,2,3,4,5,6,7,8,9,10],
                [10,9,8,7,6,5,4,3,2,1]
            ]
        }
    )
    assert response.status_code == 200
    body = response.json()
    assert "predictions" in body
    assert body["batch_size"] == 2
    assert body["output_dim"] == 1
    assert len(body["predictions"]) == 2
