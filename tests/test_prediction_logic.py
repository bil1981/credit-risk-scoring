import pytest


def test_probability_is_between_zero_and_one(client):
    response = client.post("/predict", json={"SK_ID_CURR": 100001, "features": {}})
    assert response.status_code == 200
    assert 0.0 <= response.json()["probability"] <= 1.0


def test_decision_matches_threshold(client):
    response = client.post("/predict", json={"SK_ID_CURR": 100001, "features": {}})
    assert response.status_code == 200
    result = response.json()
    expected = "REFUSED" if result["probability"] >= 0.30 else "APPROVED"
    assert result["decision"] == expected


@pytest.mark.parametrize("sk_id", [1, 100001, 123456, 999999])
def test_multiple_client_ids(client, sk_id):
    response = client.post("/predict", json={"SK_ID_CURR": sk_id, "features": {}})
    assert response.status_code == 200
    assert response.json()["SK_ID_CURR"] == sk_id
