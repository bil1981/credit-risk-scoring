# API FastAPI

L'application est definie dans `src/api.py`.

## `GET /health`

Verifie que l'API est disponible :

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Reponse attendue :

```json
{"status": "ok"}
```

## `POST /predict`

Requete minimale :

```json
{
  "SK_ID_CURR": 100001,
  "features": {
    "AMT_INCOME_TOTAL": 135000,
    "AMT_CREDIT": 568800,
    "AMT_ANNUITY": 20500,
    "DAYS_BIRTH": -12000,
    "DAYS_EMPLOYED": -2000
  }
}
```

Reponse :

```json
{
  "SK_ID_CURR": 100001,
  "probability": 0.1446,
  "decision": "APPROVED"
}
```

Le service aligne les colonnes recues sur celles du modele. Pour une prediction fiable, transmettez les features utilisees lors de l'entrainement. Les erreurs de validation renvoient `422`; les erreurs internes renvoient `500`.

## Documentation interactive

- Swagger UI : `http://127.0.0.1:8000/docs`
- ReDoc : `http://127.0.0.1:8000/redoc`
