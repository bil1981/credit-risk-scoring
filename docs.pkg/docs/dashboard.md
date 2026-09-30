# Dashboard Streamlit

Le dashboard est defini dans `app.py` et utilise la base SQLite `data/proceed/credit_risk.db`.

Lancer l'interface :

```powershell
python -m streamlit run app.py
```

## Fonctions

- recherche d'un client par `SK_ID_CURR` ;
- affichage de la probabilite de defaut et de la decision ;
- edition de variables selectionnees ;
- appel de l'API pour simuler une prediction ;
- visualisation de l'importance globale des variables.

Si la base est absente, construisez-la avec le script de preparation prevu par le projet avant de lancer Streamlit.
