# Credit Risk MLOps

Ce projet expose un modele LightGBM de prediction du risque de defaut de paiement a partir des donnees Home Credit.

## Composants

- **Modele** : LightGBM, stocke dans `models/lightgbm_model.txt`.
- **API** : FastAPI dans `src/api.py`, avec les endpoints `/health` et `/predict`.
- **Dashboard** : application Streamlit dans `app.py`.
- **Monitoring** : suivi du drift avec Evidently dans `monitoring/monitoring.py`.
- **RAG** : extraction PDF, embeddings Mistral et recherche FAISS dans `RAG/rag.py`.
- **Tests** : tests automatises dans `tests/`.

## Parcours recommande

1. Installer l'environnement avec la page [Demarrage](getting-started.md).
2. Lancer les services et verifier l'API avec [API](api.md).
3. Ouvrir le dashboard avec [Dashboard](dashboard.md).
4. Executer les tests et le monitoring avec [Tests](tests.md) et [Monitoring](monitoring.md).

## Regle de decision

L'API retourne `REFUSED` lorsque la probabilite predite est superieure ou egale a `0.30`. Sinon, elle retourne `APPROVED`.# Welcome to MkDocs

For full documentation visit [mkdocs.org](https://www.mkdocs.org).

## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
