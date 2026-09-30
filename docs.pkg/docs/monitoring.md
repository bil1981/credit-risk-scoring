# Monitoring

Le script `monitoring/monitoring.py` compare les donnees de reference et les donnees de production avec Evidently.

## Execution

Depuis la racine du projet :

```powershell
python monitoring/monitoring.py
```

Prealables :

- `logs/reference_data.csv` doit exister ;
- `logs/production_data.csv` doit contenir des predictions ;
- les deux fichiers doivent partager des colonnes de features.

Les colonnes techniques comme `timestamp`, `probability`, `decision`, `latency_ms` et `status` sont exclues du calcul de drift. Le rapport est ecrit dans `monitoring_report.html`.
