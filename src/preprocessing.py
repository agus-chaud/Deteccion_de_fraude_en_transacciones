"""
Pipeline de preprocesamiento para detección de fraude.

Mismo flujo para entrenamiento e inferencia (API):
1. Transformación de Amount: log1p(Amount)
2. Escalado con RobustScaler
3. Columnas en orden fijo: Time, V1..V28, Amount

Uso:
    from src.preprocessing import FraudPreprocessor, FEATURE_COLUMNS

    preprocessor = FraudPreprocessor()
    X_train_ready = preprocessor.fit_transform(X_train_raw)
    X_test_ready = preprocessor.transform(X_test_raw)
    # O desde la API (un solo registro):
    X_ready = preprocessor.transform({"Time": 0, "V1": -1.2, ...})
"""

from typing import Union, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


# Columnas que usa el modelo, en el orden esperado (sin Class).
# Coincide con el notebook: Time, V1..V28, Amount (Amount ya transformado con log1p internamente).
FEATURE_COLUMNS: List[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def _to_dataframe(X: Union[pd.DataFrame, dict]) -> pd.DataFrame:
    """Convierte dict (ej. body de la API) a DataFrame de una fila."""
    if isinstance(X, dict):
        return pd.DataFrame([X])
    return pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X


def _transform_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica log1p a Amount y deja el resto igual. No modifica el DataFrame in-place."""
    out = df.copy()
    if "Amount" in out.columns:
        out["Amount"] = np.log1p(out["Amount"].astype(float))
    return out


class FraudPreprocessor:
    """
    Preprocesador reutilizable: transformación de Amount (log1p) + RobustScaler.
    Acepta DataFrame o dict (una o varias filas).
    """

    def __init__(self):
        self.scaler_ = RobustScaler()
        self.feature_columns_ = FEATURE_COLUMNS.copy()

    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Selecciona y ordena las columnas esperadas. Lanza si falta alguna."""
        missing = [c for c in self.feature_columns_ if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}. Esperadas: {self.feature_columns_}")
        return df[self.feature_columns_].copy()

    def fit(self, X: Union[pd.DataFrame, dict]) -> "FraudPreprocessor":
        """
        Ajusta el escalador sobre datos crudos (con Amount sin transformar).
        X debe contener: Time, V1..V28, Amount (y opcionalmente Class, que se ignora).
        """
        df = _to_dataframe(X)
        if "Class" in df.columns:
            df = df.drop(columns=["Class"])
        df = self._ensure_columns(_transform_amount(df))
        self.scaler_.fit(df)
        return self

    def transform(self, X: Union[pd.DataFrame, dict]) -> np.ndarray:
        """
        Transforma datos crudos a features listas para el modelo.
        Si el preprocesador no fue ajustado (inferencia sin fit previo), puede fallar.
        """
        df = _to_dataframe(X)
        if "Class" in df.columns:
            df = df.drop(columns=["Class"])
        df = self._ensure_columns(_transform_amount(df))
        return self.scaler_.transform(df)

    def fit_transform(self, X: Union[pd.DataFrame, dict]) -> np.ndarray:
        """Ajusta y transforma en un solo paso (entrenamiento)."""
        return self.fit(X).transform(X)


# Documentación para la API: columnas requeridas, tipos y orden.
# La API debe aceptar un JSON con estas claves (todas numéricas).
API_FEATURE_SPEC = {
    "columns": FEATURE_COLUMNS,
    "types": {c: "float" for c in FEATURE_COLUMNS},
    "description": "Time (segundos desde la primera transacción), V1-V28 (PCA anonimizado), Amount (monto de la transacción en euros).",
}
