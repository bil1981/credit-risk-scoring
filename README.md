# Projet MLOps - Home Credit Default Risk

Ce projet met en oeuvre un modèle LightGBM de prédiction du risque de défaut de paiement à partir des données Home Credit. Il comprend :

- un pipeline de préparation et de feature engineering dans `notebooks/` ;
- un modèle LightGBM sauvegardé dans `models/lightgbm_model.txt` ;
- une base SQLite contenant les données clients, les prédictions et l'importance des variables ;
- un dashboard Streamlit pour l'exploration des résultats ;
- une API FastAPI pour exposer les prédictions ;
- une structure dédiée pour les tests et les workflows CI/CD.

## 1. Structure du projet

```text
P1_Code_Python/
├── src/
│   ├── api.py              # API FastAPI
│   ├── build_database.py   # Construction de credit_risk.db depuis des CSV
├── models/
│   └── lightgbm_model.txt
├── notebooks/
│   └── lightgbm_script.py
├── src/                    # Données CSV et fichiers de résultats
├── tests/                  # Tests automatisés API et tests unitaires/API (Pytest/Pandera)
├── .github/workflows/      # Workflows GitHub Actions, CI/CD yaml
├── monitoring/      		# rapports de dérive (Drif) Evidently AI
├── logs/            		# le fichier de sauvegarde de données de production (production_data.csv)
├── docs/            		# documentation mkdocs
├── main.py             	# code principal lançant l'API et l'interface Streamlit
├── app.py              	# Interface/Dashboard Streamlit
├── CHANGELOG.md			# fichier Markdown qui liste chronologiquement tous les changements apportés au fil de ses version
├── Dockerfile				# liste d'instructions pour créer automatiquement une image
├── .gitignore			 	# liste des fichiers/dossiers ignorés dans le contrôle de version
├── requirements.txt		# liste l'ensemble des bibliothèques et packages externes nécessaires
└── README.md

```

## 2. Prérequis

- Windows avec Python 3.10 ou version supérieure ;
- les données Home Credit dans `src/` ;
- le modèle `models/lightgbm_model.txt` ;
- PowerShell ou un terminal VS Code.

Les chemins utilisés par les scripts sont relatifs à la racine du projet. Toutes les commandes suivantes doivent donc être exécutées depuis `P1_Code_Python`.

## 3. Installation de l'environnement

Créer un environnement virtuel :

```powershell
python -m venv .venv
```

Activer l'environnement dans PowerShell :

```powershell
.\.venv\Scripts\Activate.ps1
```

Installer les dépendances :

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Les dépendances principales sont `pandas`, `numpy`, `lightgbm`, `streamlit`, `plotly`, `fastapi`, `uvicorn`, `pydantic`, `pytest` et `httpx`.

## 4. Préparation des données et du modèle

Le script de modélisation situé dans `notebooks/lightgbm_script.py` lit les fichiers suivants :

- `Data/raw//application_train.csv` ;
- `Data/raw/application_test.csv` ;
- `Data/raw/bureau.csv` et `src/bureau_balance.csv` ;
- `Data/raw/previous_application.csv` ;
- `Data/raw/POS_CASH_balance.csv` ;
- `Data/raw/installments_payments.csv` ;
- `Data/raw/credit_card_balance.csv`.

Le pipeline effectue notamment le one-hot encoding, les agrégations par `SK_ID_CURR`, le feature engineering et l'entraînement LightGBM avec validation croisée. Le modèle final doit être disponible sous :

```text
models/lightgbm_model.txt
```

Les fichiers de prédictions et d'importance des variables utilisés par la base sont notamment :

```fichier  csv de resultats Lightgbm
input/submission_kernel02.csv
input/feature_importance.csv
```

## 5. Construction de la base SQLite

Construire la base utilisée par le dashboard :

```powershell
python .\app\build_database.py
```

Cette commande crée ou remplace `credit_risk.db` et génère principalement :

- `client_features` : données clients fusionnées avec les prédictions ;
- `feature_importance` : importance moyenne des variables.


## 6. Lancer le dashboard Streamlit

Depuis la racine du projet :

```powershell
streamlit run app.py
```

Le dashboard permet de :

- rechercher un client par `SK_ID_CURR` ;
- consulter sa probabilité de défaut et sa décision ;
- afficher et modifier des variables sélectionnées ;
- consulter l'importance globale des variables.

La règle métier utilisée dans l'interface est :

- probabilité `< 0.30` : `APPROVED` ;
- probabilité `>= 0.30` : `REFUSED`.

## 7. Lancer l'API FastAPI

Démarrer le serveur de développement :

```powershell
uvicorn app.main:app --reload
```

L'API est disponible à l'adresse `http://127.0.0.1:8000`.

Documentation interactive :

- Swagger UI : `http://127.0.0.1:8000/docs` ;
- ReDoc : `http://127.0.0.1:8000/redoc`.

Au démarrage, le cycle de vie FastAPI charge une seule fois le modèle LightGBM et la connexion SQLite. Ces ressources sont libérées à l'arrêt de l'application.

### Vérifier l'état de l'API

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health -Method Get
```

Réponse attendue :

```json
{"status":"ok"}
```

### Appeler `/predict`

Le payload doit contenir `SK_ID_CURR` et un objet `features`. Les noms et valeurs doivent correspondre aux 238 features attendues par le modèle. Les features sont numériques, y compris les variables catégorielles déjà encodées et les colonnes issues du one-hot encoding.

Exemple de structure :

```json
{
	"SK_ID_CURR": 100001,
	"features": {
		"NAME_CONTRACT_TYPE": 0,
		"FLAG_OWN_CAR": 1,
		"AMT_INCOME_TOTAL": 180000.0,
		"AMT_CREDIT": 500000.0,
		"EXT_SOURCE_2": 0.65
	}
}
```

L'exemple ci-dessus est volontairement abrégé. Pour obtenir une prédiction, toutes les features attendues par le modèle doivent être fournies. Une feature manquante ou inconnue renvoie une erreur HTTP `422` avec le détail de validation.

Exemple PowerShell avec un fichier JSON complet :

```powershell
Invoke-RestMethod `
	-Uri http://127.0.0.1:8000/predict `
	-Method Post `
	-ContentType "application/json" `
	-InFile .\payload.json
```

Réponse :

```json
{
	"SK_ID_CURR": 100001,
	"probability": 0.1446,
	"decision": "REFUSED"
}
```

## 8. Tests et validations

Lancer les tests présents dans le projet :

```powershell
pytest
```

Les vérifications manuelles effectuées lors de la mise en place sont :

```powershell
python -m py_compile .\app\main.py
python .\app\check_db.py
```

L'API a également été vérifiée avec `TestClient` pour les cas suivants :

- `/health` renvoie `200` et `status: ok` lorsque le modèle est chargé ;
- `/predict` renvoie une probabilité et une décision ;
- un payload incomplet renvoie `422`.

## 9. Déroulement de la mise en place

1. Création ou vérification des répertoires `app/`, `tests/`, `.github/` et `.github/workflows/`.
2. Création de `requirements.txt` avec les dépendances data science, Streamlit, API et tests.
3. Identification du modèle LightGBM existant et de ses features attendues.
4. Création de `app/main.py` avec les schémas Pydantic.
5. Ajout du gestionnaire `lifespan` pour charger le modèle et SQLite au démarrage.
6. Ajout des routes `/health` et `/predict`.
7. Validation des features manquantes et inconnues avant prédiction.
8. Ajout de `lightgbm` aux dépendances pour rendre le déploiement reproductible.
9. Compilation et tests fonctionnels de l'API.
10. Documentation de l'installation, de la base, du dashboard et de l'API dans ce guide.

## 10. Dépannage

### Modèle introuvable

Vérifier que `models/lightgbm_model.txt` existe et lancer `uvicorn` depuis la racine du projet.

### Base SQLite introuvable ou vide

Exécuter `python .\app\build_database.py`, puis contrôler le résultat avec `python .\app\check_db.py`.

### Erreur `422` sur `/predict`

Le payload ne contient pas toutes les colonnes attendues ou contient une colonne inconnue. Utiliser les noms exacts des features du modèle et fournir une valeur numérique pour chacune.

### Port déjà utilisé

Lancer FastAPI sur un autre port :

```powershell
uvicorn app.main:app --reload --port 8001
```

## Assistant RAG Mistral

Le sous-projet `P10_DSML` contient un assistant RAG basé sur FAISS et Mistral ainsi qu'un script d'évaluation RAGAS. Depuis la racine du dépôt, entrez dans le dossier du sous-projet, puis créez l'environnement et installez ses dépendances :

```powershell
Set-Location .\P10_DSML
uv venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Créez un fichier `.env` dans `P10_DSML` et configurez votre clé API sans la publier :

```text
MISTRAL_API_KEY=votre_clé_api_mistral
```

Vérifiez que `vector_db/faiss_index.idx` et `vector_db/document_chunks.pkl` existent; sinon, lancez l'indexation avec `uv run python indexer.py`.

Lancer l'évaluation :

```powershell
uv run python evaluate_ragas.py
```

Le script calcule `faithfulness`, `answer_relevancy`, `context_precision` et `context_recall`, puis enregistre les résultats dans `ragas_evaluation_results.csv`. Il envoie des requêtes à l'API Mistral, susceptibles d'être facturées et soumises à des quotas. La concurrence RAGAS est limitée à un worker. En cas de réponse HTTP `429`, vérifiez les limites de débit et les crédits du compte avant de relancer.

### arcitecture stramlit et Fastapi

[ Utilisateur ]
         │
         ▼ (Saisit les données sur l'écran)
┌──────────────────────────────────────────────────────────┐
│              Streamlit (app_2.py) - Front                │
└──────────────────────────┬───────────────────────────────┘
                           │
                           │ Requête HTTP POST (JSON)
                           │ ex: http://localhost:8000/predict
                           ▼
┌──────────────────────────────────────────────────────────┐
│              FastAPI (main.py) - Back                    │
│  - Reçoit les données                                    │
│  - Exécute le modèle LightGBM chargé au démarrage       │
│  - Renvoie probabilité + décision                        │
└──────────────────────────┬───────────────────────────────┘
                           │
                           │ Réponse HTTP (JSON)
                           │ ex: {"probability": 0.15, "decision": "REFUSED"}
                           ▼
┌──────────────────────────────────────────────────────────┐
│              Streamlit (app_2.py) - Front                │
│  - Affiche la metric et le message de succès/erreur      │
└──────────────────────────────────────────────────────────┘
