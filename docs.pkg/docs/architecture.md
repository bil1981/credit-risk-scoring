# Architecture

```text
Donnees Home Credit
        |
        v
Feature engineering et entrainement LightGBM
        |
        +--> models/lightgbm_model.txt
        |
        +--> src/api.py --> /health, /predict --> logs/production_data.csv
        |
        +--> app.py --> dashboard Streamlit
        |
        +--> monitoring/monitoring.py --> monitoring_report.html
```

## Flux de prediction

1. FastAPI charge le modele au demarrage.
2. `/predict` recoit `SK_ID_CURR` et les features du client.
3. Les variables categorielles sont encodees avec `get_dummies`.
4. Les colonnes sont alignees sur les features attendues par LightGBM.
5. La probabilite est calculee et convertie en decision.
6. La prediction et sa latence sont ajoutees au journal de production.

## Artefacts importants

- `models/lightgbm_model.txt` : modele entraine.
- `models/feature_names.joblib` : noms de variables disponibles.
- `logs/reference_data.csv` : donnees de reference pour le drift.
- `logs/production_data.csv` : donnees issues des predictions.
- `monitoring_report.html` : rapport Evidently genere.
