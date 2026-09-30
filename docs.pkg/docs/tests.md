# Tests et validation

Lancer la suite de tests :

```powershell
pytest
```

Les tests couvrent notamment :

- la presence et le titre de l'application FastAPI ;
- le fonctionnement de `/predict` ;
- les champs obligatoires et les erreurs de type ;
- la logique de prediction ;
- les schemas Pandera et la base de donnees.

Verifications rapides :

```powershell
python -m py_compile src/api.py
python -m py_compile app.py
Invoke-RestMethod http://127.0.0.1:8000/health
```

Pour tester l'API sans demarrer un serveur, utilisez le `TestClient` configure dans `tests/conftest.py`.
