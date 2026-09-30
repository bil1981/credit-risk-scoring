# Tests pytest - Credit Risk

validation/schemas.py : Définit la règle de gestion et la structure attendue des données de crédit.

test_validation_donnees.py (Test Unitaire) : Vérifie que la règle de gestion de schemas.py fonctionne correctement en Python pur.

test_api.py (Test d'Intégration) : Vérifie que l'API web intègre correctement la règle et renvoie le bon code de statut HTTP (422) au client.

test_prediction_logic.py est un jeu de tests d'intégration ciblés sur les règles de décision du modèle et de l'API. Alors que test_pandera.py vérifie la qualité des données d'entrée et test_api.py vérifie la structure HTTP globale, ce fichier valide le comportement fonctionnel de la prédiction.

test_streamlit.py permet de test la presence du script app

test_database.py test l'accès a la base de donnérs


Depuis la racine du projet :

```bash
pip install -r tests/requirements-test.txt
pytest -v
```

API uniquement :

```bash
pytest tests/test_main.py -v
```

Logique de prédiction :

```bash
pytest tests/test_prediction_logic.py -v
```

Base SQLite :

```bash
pytest tests/test_database.py -v
```

Le modèle `models/lightgbm_model.txt` doit être présent pour les tests API.
La base `credit_risk.db` doit être présente pour les tests SQLite.
