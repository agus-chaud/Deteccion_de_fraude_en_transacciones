"""
API REST para detección de fraude.

Ejecutar desde la raíz del proyecto:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health  -> 200 (servicio vivo)
    POST /predict -> body: transacción (Time, V1..V28, Amount); response: is_fraud, score, version
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Asegurar que la raíz del proyecto esté en el path (scripts, src, models)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException

from api.schemas import PredictResponse, TransactionInput


def _load_app_model(version: str = "v1"):
    """Carga modelo, preprocesador y metadata. Usado al arranque."""
    from scripts.export_model import load_model
    return load_model(version=version)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Al arrancar: cargar modelo, preprocesador y metadata."""
    version = "v1"
    try:
        model, preprocessor, metadata = _load_app_model(version=version)
        app.state.model = model
        app.state.preprocessor = preprocessor
        app.state.threshold = float(metadata["threshold"])
        app.state.version = metadata.get("version", version)
    except FileNotFoundError as e:
        raise RuntimeError(f"No se pudo cargar el modelo '{version}'. ¿Ejecutaste el notebook y export_model? {e}") from e
    yield
    # shutdown: nada que cerrar


app = FastAPI(
    title="API Detección de Fraude",
    description="Recibe una transacción y devuelve si es fraude y el score.",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Devuelve 200 si el servicio está vivo (para Docker/orquestadores)."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(body: TransactionInput):
    """
    Recibe una transacción (JSON con Time, V1..V28, Amount).
    Devuelve is_fraud (true/false) y score (probabilidad de fraude).
    """
    # Pydantic ya validó el body; convertimos a dict para el preprocesador
    raw = body.model_dump()

    preprocessor = app.state.preprocessor
    model = app.state.model
    threshold = app.state.threshold
    version = getattr(app.state, "version", None)

    try:
        X = preprocessor.transform(raw)
        score = float(model.predict_proba(X)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error al predecir: {e!s}") from e

    is_fraud = score >= threshold

    return PredictResponse(is_fraud=is_fraud, score=round(score, 6), version=version)
