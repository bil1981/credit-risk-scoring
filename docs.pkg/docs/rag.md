# Module RAG

Le script `RAG/rag.py` transforme un PDF municipal en assistant de question-reponse.

## Pipeline

1. Extraction du texte avec PyMuPDF et OCR Tesseract si necessaire.
2. Decoupage en blocs de 500 caracteres avec 100 caracteres de recouvrement.
3. Creation des embeddings avec le modele Mistral `mistral-embed`.
4. Indexation des vecteurs avec FAISS.
5. Recherche des trois blocs les plus proches.
6. Generation d'une reponse avec `mistral-small-latest`.

## Configuration

La cle ne doit pas etre ecrite dans le code. Dans PowerShell :

```powershell
$env:MISTRAL_API_KEY = "votre_cle_mistral"
python RAG/rag.py
```

Une erreur `401` indique une cle absente, invalide ou revoquee. Une erreur `429` indique une limite de frequence ou de quota Mistral; le script effectue quelques nouvelles tentatives, mais ne peut pas contourner un quota epuise.

Le PDF attendu par l'exemple est `RAG/reglement_municipale.pdf`.
