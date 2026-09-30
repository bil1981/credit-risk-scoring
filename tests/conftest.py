import pytest
from fastapi.testclient import TestClient
from src.api import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


#
# Cla permet à tous les tests de faire simplement :
#
# def test_predict(client):
#  response = client.post(...)

# s #ans recréer le TestClient à chaque fois.
