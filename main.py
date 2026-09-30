import subprocess
import sys
import time


def run_services():
    print("Démarrage de l'API FastAPI et de Streamlit...")

    # 1. Démarrer FastAPI (api.py)
    api_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--reload",
        ]
    )

    # Laisser 2 secondes à l'API pour s'initialiser
    time.sleep(2)

    # 2. Démarrer Streamlit (app.py)
    streamlit_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"]
    )

    try:
        # Maintenir le script actif
        api_process.wait()
        streamlit_process.wait()
    except KeyboardInterrupt:
        print("\n Arrêt des services...")
        api_process.terminate()
        streamlit_process.terminate()
        print("Services arrêtés avec succès.")


if __name__ == "__main__":
    run_services()
