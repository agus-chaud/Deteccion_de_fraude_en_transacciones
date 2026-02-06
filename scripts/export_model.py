"""
Script de exportación del modelo y preprocesador a una versión en models/.

Uso desde el notebook (recomendado la primera vez):

    from scripts.export_model import export_model

    export_model(
        model=model_xgb,           # el modelo entrenado (ej. XGBoost, LightGBM)
        preprocessor=preprocessor,  # el FraudPreprocessor ya guardado o el que usaste en la celda de validación
        version="v1",
        X_test=X_test,             # DataFrame o array ya preprocesado (mismo que usaste para evaluar)
        y_test=y_test,             # Series o array con las etiquetas reales
        threshold=0.5,
    )

Eso escribe en models/v1/: model.joblib, preprocessor.joblib, metadata.json.

Cargar una versión (para la API o para probar):

    from scripts.export_model import load_model

    model, preprocessor, metadata = load_model(version="v1")
    threshold = metadata["threshold"]
    # Predecir: X_ready = preprocessor.transform(datos_crudos); score = model.predict_proba(X_ready)[:, 1]; is_fraud = score >= threshold

Uso por línea de comandos (cuando ya tenés modelo y preprocesador en archivos):

    python scripts/export_model.py --version v1 --model path/to/model.joblib --preprocessor path/to/preprocessor.joblib
    python scripts/export_model.py --version v1 --model m.joblib --preprocessor p.joblib --X-test X_test.csv --y-test y_test.csv --threshold 0.5
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

# Métricas (solo si tenemos y_test y predicciones)
def _compute_metrics(y_true, y_pred, y_scores):
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_scores)),
        "aupr": float(average_precision_score(y_true, y_scores)),
    }


def export_model(
    model,
    preprocessor,
    version: str,
    *,
    X_test=None,
    y_test=None,
    threshold: float = 0.5,
):
    """
    Guarda modelo, preprocesador y metadata en models/{version}/.

    Parámetros
    ----------
    model : objeto con .predict() y .predict_proba()
        Modelo entrenado (ej. XGBoost, LightGBM).
    preprocessor : FraudPreprocessor
        Preprocesador ya ajustado (fit en train).
    version : str
        Nombre de la versión, ej. "v1", "v2".
    X_test : array o DataFrame, opcional
        Conjunto de test ya preprocesado (mismas columnas que usa el modelo).
        Si se pasa junto con y_test, se calculan métricas en test.
    y_test : array o Series, opcional
        Etiquetas reales del test. Debe tener el mismo orden que X_test.
    threshold : float
        Umbral de probabilidad para clasificar como fraude (default 0.5).
        Se guarda en metadata para que la API lo use en /predict.
    """
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "models" / version
    out_dir.mkdir(parents=True, exist_ok=True)

    # Columnas que espera el modelo (del preprocesador)
    feature_columns = getattr(preprocessor, "feature_columns_", None)
    if feature_columns is None:
        from src.preprocessing import FEATURE_COLUMNS
        feature_columns = FEATURE_COLUMNS

    # Métricas en test (si nos pasan X_test e y_test)
    metrics = {}
    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        try:
            y_scores = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_scores = y_pred.astype(float)
        y_true = np.asarray(y_test)
        metrics = _compute_metrics(y_true, y_pred, y_scores)

    metadata = {
        "version": version,
        "threshold": threshold,
        "metrics": metrics,
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "feature_columns": list(feature_columns),
    }

    joblib.dump(model, out_dir / "model.joblib")
    joblib.dump(preprocessor, out_dir / "preprocessor.joblib")
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Exportado en {out_dir}: model.joblib, preprocessor.joblib, metadata.json")
    if metrics:
        print("Métricas en test:", metrics)
    return out_dir


def load_model(version: str = "v1"):
    """
    Carga modelo, preprocesador y metadata desde models/{version}/.

    Parámetros
    ----------
    version : str
        Nombre de la versión, ej. "v1", "v2".

    Returns
    -------
    model : modelo cargado (joblib)
    preprocessor : FraudPreprocessor cargado
    metadata : dict con "version", "threshold", "metrics", "exported_at", "feature_columns"

    Raises
    ------
    FileNotFoundError
        Si no existe la carpeta models/{version}/ o alguno de los archivos.
    """
    root = Path(__file__).resolve().parent.parent
    version_dir = root / "models" / version

    if not version_dir.is_dir():
        raise FileNotFoundError(f"No existe la versión '{version}' en {version_dir}")

    model_path = version_dir / "model.joblib"
    preprocessor_path = version_dir / "preprocessor.joblib"
    metadata_path = version_dir / "metadata.json"

    for p, name in [(model_path, "model.joblib"), (preprocessor_path, "preprocessor.joblib"), (metadata_path, "metadata.json")]:
        if not p.exists():
            raise FileNotFoundError(f"Falta {name} en {version_dir}")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return model, preprocessor, metadata


def _main():
    parser = argparse.ArgumentParser(description="Exportar modelo y preprocesador a models/{version}/")
    parser.add_argument("--version", default="v1", help="Versión, ej. v1")
    parser.add_argument("--model", required=True, help="Ruta al model.joblib (o archivo del modelo)")
    parser.add_argument("--preprocessor", required=True, help="Ruta al preprocessor.joblib")
    parser.add_argument("--X-test", help="CSV con features de test (opcional, para calcular métricas)")
    parser.add_argument("--y-test", help="CSV con una columna de etiquetas (opcional)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Umbral de decisión")
    args = parser.parse_args()

    model = joblib.load(args.model)
    preprocessor = joblib.load(args.preprocessor)

    X_test = None
    y_test = None
    if args.X_test and args.y_test:
        import pandas as pd
        X_test = pd.read_csv(args.X_test)
        y_test = pd.read_csv(args.y_test).iloc[:, 0]
    elif args.X_test or args.y_test:
        print("Advertencia: se ignoran --X-test/--y-test si no se pasan ambos.")

    export_model(
        model=model,
        preprocessor=preprocessor,
        version=args.version,
        X_test=X_test,
        y_test=y_test,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    _main()
