from src.api import app


# Vérifie que l'application FastAPI est correctement créée.
def test_app_exists():
    assert app is not None
    assert app.title == "Credit Risk API"


# Vérifie qu'une requête complète retourne une prédiction valide.
def test_predict_endpoint_returns_valid_response(client):
    # Prépare un jeu de données contenant les principales variables de crédit.
    payload = {
        "SK_ID_CURR": 100001,
        "features": {
            "AMT_INCOME_TOTAL": 135000,
            "AMT_CREDIT": 568800,
            "AMT_ANNUITY": 20500,
            "DAYS_BIRTH": -12000,
            "DAYS_EMPLOYED": -2000,
        },
    }
    response = client.post("/predict", json=payload)
    # Une requête valide doit retourner le statut HTTP 200.
    assert response.status_code == 200
    data = response.json()
    # Vérifie l'identifiant retourné par l'API.
    assert data["SK_ID_CURR"] == 100001
    # Vérifie que la probabilité est numérique et comprise entre 0 et 1.
    assert isinstance(data["probability"], float)
    assert 0.0 <= data["probability"] <= 1.0
    # Vérifie que la décision correspond à une valeur autorisée.
    assert data["decision"] in {"APPROVED", "REFUSED"}


# Vérifie que l'identifiant client est obligatoire dans la requête.
def test_predict_requires_sk_id_curr(client):
    assert client.post("/predict", json={"features": {}}).status_code == 422


# Vérifie que la réponse contient exactement les champs attendus.
def test_predict_response_contains_expected_fields(client):
    response = client.post("/predict", json={"SK_ID_CURR": 100001, "features": {}})
    assert response.status_code == 200
    assert set(response.json()) == {"SK_ID_CURR", "probability", "decision"}


# Vérifie qu'un identifiant de type incorrect est rejeté par Pydantic.
def test_predict_rejects_invalid_sk_id_curr_type(client):
    response = client.post("/predict", json={"SK_ID_CURR": "abc", "features": {}})
    assert response.status_code == 422


# Vérifie que Pandera rejette un montant de crédit négatif.
def test_predict_rejects_invalid_credit_amount(client):
    response = client.post(
        "/predict",
        json={
            "SK_ID_CURR": 100001,
            "features": {"AMT_CREDIT": -1},
        },
    )

    # Les erreurs de validation des données sont retournées avec le statut 422.
    assert response.status_code == 422
    # Vérifie que l'erreur provient bien de la validation Pandera.
    assert "Validation des données échouée" in response.json()["detail"]
