from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict

import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from validation.schemas import validate_credit_data


# ============================================================
# CHEMINS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_model.txt"

LOG_DIR = PROJECT_ROOT / "logs"
PRODUCTION_LOG_PATH = LOG_DIR / "production_data.csv"

DECISION_THRESHOLD = 0.30


# ============================================================
# MODELES PYDANTIC
# ============================================================


class PredictionRequest(BaseModel):
    SK_ID_CURR: int
    features: Dict[str, Any] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    SK_ID_CURR: int
    probability: float
    decision: str


# ============================================================
# LOGGING CSV
# ============================================================


def log_prediction(
    payload: PredictionRequest,
    probability: float,
    decision: str,
    latency_ms: float,
    status: int = 200,
) -> None:
    """
    Ajoute une prédiction dans production_data.csv.
    Le logging ne doit jamais bloquer la prédiction.
    """

    try:
        LOG_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Construire la ligne
        # ----------------------------------------------------

        log_row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "SK_ID_CURR": "999999999",
            **payload.features,
            "probability": probability,
            "decision": decision,
            "latency_ms": round(latency_ms, 2),
            "status": status,
        }

        new_row = pd.DataFrame([log_row])

        # ----------------------------------------------------
        # CAS 1 :
        # fichier inexistant OU fichier vide
        # ----------------------------------------------------

        if not PRODUCTION_LOG_PATH.exists() or PRODUCTION_LOG_PATH.stat().st_size == 0:
            new_row.to_csv(
                PRODUCTION_LOG_PATH,
                index=False,
            )

            print(f"Production log créé : {PRODUCTION_LOG_PATH}")

            return

        # ----------------------------------------------------
        # CAS 2 :
        # fichier déjà alimenté
        # ----------------------------------------------------

        existing_columns = pd.read_csv(
            PRODUCTION_LOG_PATH,
            nrows=0,
        ).columns.tolist()

        # Respecter les colonnes existantes
        new_row = new_row.reindex(
            columns=existing_columns,
        )

        new_row.to_csv(
            PRODUCTION_LOG_PATH,
            mode="a",
            header=False,
            index=False,
        )

    except Exception as error:
        # IMPORTANT :
        # une erreur de logging ne doit jamais
        # provoquer une erreur FastAPI 500

        print(f"Erreur logging CSV : {error}")


# ============================================================
# LIFESPAN
# ============================================================


@asynccontextmanager  # il sert à gérer le cycle de vie de l'application FastAPI
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Modèle introuvable : {MODEL_PATH}")

    # Charger le modèle LightGBM
    app.state.model = lgb.Booster(model_file=str(MODEL_PATH))

    # Utiliser les noms des variables réellement enregistrés dans le modèle
    app.state.feature_names = list(app.state.model.feature_name())

    print(f" {len(app.state.feature_names)} variables chargées depuis LightGBM")

    print(f"Logging production : {PRODUCTION_LOG_PATH}")

    yield


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Credit Risk API",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# PREDICTION
# ============================================================


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    payload: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    start_time = perf_counter()

    model = request.app.state.model
    feature_names = request.app.state.feature_names

    try:
        # ====================================================
        # 1. DONNEES RECUES
        # ====================================================

        df = pd.DataFrame([payload.features])

        df["SK_ID_CURR"] = payload.SK_ID_CURR

        try:
            df = validate_credit_data(df)
        except Exception as error:
            raise HTTPException(
                status_code=422,
                detail=(f"Validation des données échouée : {error}"),
            ) from error

        raw_df = df

        # ====================================================
        # 2. SUPPRESSION TARGET / DECISION
        # ====================================================

        raw_df = raw_df.drop(
            columns=[
                "TARGET",
                "DECISION",
            ],
            errors="ignore",
        )

        # ====================================================
        # 3. VARIABLES CATEGORIELLES
        # ====================================================

        categorical_columns = raw_df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        encoded_df = pd.get_dummies(
            raw_df,
            columns=categorical_columns,
            dummy_na=False,
        )

        # ====================================================
        # 4. ALIGNEMENT AVEC LIGHTGBM
        # ====================================================

        encoded_df = encoded_df.reindex(
            columns=feature_names,
            fill_value=0,
        )

        # ====================================================
        # 5. CONVERSION NUMERIQUE
        # ====================================================

        encoded_df = encoded_df.apply(
            pd.to_numeric,
            errors="coerce",
        ).fillna(0)

        # ====================================================
        # 6. VERIFICATION
        # ====================================================

        if encoded_df.shape[1] != model.num_feature():
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Nombre de variables incorrect : "
                    f"{encoded_df.shape[1]} reçues, "
                    f"{model.num_feature()} attendues."
                ),
            )

        # ====================================================
        # 7. PREDICTION
        # ====================================================

        probability = float(model.predict(encoded_df)[0])

        decision = "REFUSED" if probability >= DECISION_THRESHOLD else "APPROVED"

        # ====================================================
        # 8. LATENCE
        # ====================================================

        latency_ms = (perf_counter() - start_time) * 1000

        # ====================================================
        # 9. LOG PRODUCTION
        # ====================================================

        log_prediction(
            payload=payload,
            probability=probability,
            decision=decision,
            latency_ms=latency_ms,
            status=200,
        )

        # ====================================================
        # 10. REPONSE
        # ====================================================

        return PredictionResponse(
            SK_ID_CURR=payload.SK_ID_CURR,
            probability=probability,
            decision=decision,
        )

    except HTTPException:
        raise

    except Exception as error:
        latency_ms = (perf_counter() - start_time) * 1000

        # Logger l'erreur,
        log_prediction(
            payload=payload,
            probability=None,
            decision="ERROR",
            latency_ms=latency_ms,
            status=500,
        )

        raise HTTPException(
            status_code=500,
            detail=(f"Erreur lors de la prédiction : {error}"),
        )
