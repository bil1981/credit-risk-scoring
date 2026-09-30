# Demarrage

Toutes les commandes se lancent depuis la racine `P1_Code_Python`.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Le projet utilise notamment Python, pandas, NumPy, LightGBM, FastAPI, Streamlit, FAISS, Evidently et pytest.

## Lancer les services

Pour lancer l'API et le dashboard ensemble :

```powershell
python main.py
```

Adresses locales :

- API : `http://127.0.0.1:8000`
- Swagger : `http://127.0.0.1:8000/docs`
- Dashboard : `http://127.0.0.1:8501`
- Documentation MkDocs : `http://127.0.0.1:8002`

Lancement separe :

```powershell
python -m uvicorn src.api:app --reload
python -m streamlit run app.py
```

Lancer MkDocs sur un port distinct :

```powershell
python -m mkdocs serve -f doc_projets2_mlops.pkg\mkdocs.yml --dev-addr 127.0.0.1:8002
```

## Docker

```powershell
docker build -t credit-risk-mlops .
docker run --rm -p 8000:8000 -p 8501:8501 credit-risk-mlops
```

Le conteneur demarre `main.py`, qui lance les deux services.
