FROM python:3.12-slim

WORKDIR /app

# Installation des dépendances système si nécessaire
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copie du reste du code projet
COPY . .

# Exposition des ports FastAPI (8000) et Streamlit (8501)
EXPOSE 8000 8501

# Commande de démarrage
CMD ["python", "main.py"]
