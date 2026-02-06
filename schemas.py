"""
Modelos Pydantic para request y response de la API.

El body de POST /predict debe tener las mismas columnas que el modelo:
Time, V1..V28, Amount (todos float).
"""

from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    """Una transacción: features en bruto (mismas que en entrenamiento)."""

    Time: float = Field(..., description="Segundos desde la primera transacción")
    V1: float = Field(..., description="Feature PCA anonimizada")
    V2: float = Field(...)
    V3: float = Field(...)
    V4: float = Field(...)
    V5: float = Field(...)
    V6: float = Field(...)
    V7: float = Field(...)
    V8: float = Field(...)
    V9: float = Field(...)
    V10: float = Field(...)
    V11: float = Field(...)
    V12: float = Field(...)
    V13: float = Field(...)
    V14: float = Field(...)
    V15: float = Field(...)
    V16: float = Field(...)
    V17: float = Field(...)
    V18: float = Field(...)
    V19: float = Field(...)
    V20: float = Field(...)
    V21: float = Field(...)
    V22: float = Field(...)
    V23: float = Field(...)
    V24: float = Field(...)
    V25: float = Field(...)
    V26: float = Field(...)
    V27: float = Field(...)
    V28: float = Field(...)
    Amount: float = Field(..., description="Monto de la transacción")

    model_config = {"extra": "forbid"}  # no permitir campos extra


class PredictResponse(BaseModel):
    """Respuesta de POST /predict."""

    is_fraud: bool = Field(..., description="True si la transacción se clasifica como fraude")
    score: float = Field(..., description="Probabilidad de fraude (0-1)")
    version: str | None = Field(None, description="Versión del modelo usada (ej. v1)")
