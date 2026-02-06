# Detección de fraude en transacciones

Proyecto de ML que entrena un clasificador para detectar transacciones fraudulentas a partir de datos anonimizados (PCA) y expone la predicción vía **API REST** con FastAPI. Incluye EDA, preprocesamiento reutilizable, versionado del modelo y opción de despliegue con Docker.

---

## 1. Contexto del negocio

**Problema:** Las entidades financieras y comercios necesitan identificar en tiempo (casi) real las transacciones con tarjeta que son fraudulentas, para bloquearlas o pedir verificación y así reducir pérdidas y mejorar la confianza del usuario.

El fraude genera pérdidas directas, costos operativos (controversias, soporte) y daño reputacional. Automatizar la detección permite escalar la revisión y priorizar los casos más sospechosos en lugar de revisar todo a mano.

**Objetivo del proyecto:** Construir un modelo que, dado un registro de transacción (Time, componentes PCA anonimizados, Amount), devuelva si es fraude o no y un score de probabilidad, y exponerlo como API para poder integrarlo en un flujo de decisión (alertas, bloqueos, cola de revisión).

---

## 2. Enfoque y decisiones técnicas

**Metodología y pipeline**

- **EDA:** Análisis de distribución de clases, `Time`, `Amount` y correlaciones con `Class`; visualizaciones (barras, histogramas, boxplots, heatmap, Q-Q). Se detecta desbalance extremo (~0,17% fraude) y fuerte asimetría en `Amount`.
- **Preprocesamiento:** Separación features/target; **log1p(Amount)** para reducir asimetría; **RobustScaler** sobre todas las variables (incl. Time y Amount). Train/test estratificado 80/20. Pipeline encapsulado en `src/preprocessing.py` con **fit solo en train** para evitar data leakage en producción.
- **Desbalance:** Se comparan sin balanceo, y RandomUnderSampler (ratio ~1:3 fraude/normal). Se probó SMOTE (aumenta los fraudes), pero los modelos tardaban mas de 4 horas en correrse .
- **Modelado:** Se entrenan y evalúan Random Forest , **XGBoost** , LightGBM  e Isolation Forest. Se elige **XGBoost** como modelo final, usando el AUPR en test como métrica principal.
- **Producción:** Versionado en `models/v1/` (modelo, preprocesador, metadata); `export_model` y `load_model(version)`; API FastAPI con `/health` y `/predict`, validación Pydantic y carga del modelo al arranque; Dockerfile para contenedor.

**Justificación de decisiones**

- **AUPR como métrica principal:** Con ~0,17% de positivos, el ROC-AUC puede ser alto solo por clasificar bien la mayoría (negativos). El AUPR se centra en la clase fraude y no se infla por los TN, por eso se usa para elegir y optimizar el modelo.
- **RobustScaler:** Los montos y el tiempo pueden tener outliers; RobustScaler (mediana e IQR) es más estable que StandardScaler.
- **Fit del preprocesador solo en train:** En producción no se dispone del “futuro” test; ajustar el scaler solo con train garantiza coherencia entre entrenamiento e inferencia y evita data leakage.

---

## 3. Resultados e impacto

**Métricas en conjunto de test (modelo v1 exportado)**

| Métrica   | Valor   |
|----------|---------|
| Accuracy | 0,9995  |
| Precision| 0,8901  |
| Recall   | 0,8265  |
| F1       | 0,8571  |
| ROC-AUC  | 0,9752  |
| AUPR     | 0,8754  |

**Visualizaciones:** El notebook incluye gráficos de distribución de clases, histogramas de `Time` y `Amount`, boxplots por clase, heatmap de correlaciones con `Class` y curvas tras el preprocesamiento. 

**Impacto resumido:** El sistema detecta alrededor de **83% de los fraudes** (recall) con **89% de precision** en las alertas (en test), y un AUPR de **0,88**, lo que indica buena capacidad para rankear casos sospechosos. La API permite integrar esta predicción en un flujo operativo (por ejemplo, marcar transacciones para revisión o bloqueo según umbral y políticas del negocio).

---

## 4. Desafíos y soluciones

| Desafío | Solución |
|--------|----------|
| **Desbalance extremo (~577:1)** | Uso de AUPR y recall para evaluar; undersampling en train (ratio 1:3); en XGBoost, `scale_pos_weight` para ponderar la clase fraude. Así el modelo no se sesga a predecir siempre “normal”. |
| **Coherencia entre entrenamiento y producción** | Inicialmente el scaler se ajustaba con todo el dataset antes del split (data leakage). Se extrajo un pipeline que hace fit solo en train (`src/preprocessing.py`) y se reentrenó el modelo con datos preprocesados por ese mismo pipeline; la API usa el mismo preprocesador guardado. |
| **Métrica engañosa (ROC-AUC con mayoría negativa)** | Se adoptó AUPR como métrica principal para selección y optimización del modelo, y se documentó en el notebook por qué en este contexto el ROC-AUC puede ser engañoso. |

---

## 5. Note future improvements (mejoras futuras)

- **Despliegue en producción:** Poner la API en un entorno estable (cloud o on-prem) usando Docker y FastAPI; configurar HTTPS, límites de tasa y health checks para orquestadores.
- **Monitoreo:** Logs de predicciones, métricas de latencia y detección de drift (cambios en la distribución de las features o en la tasa de positivos).
- **Umbral y costos:** Ajustar el umbral de decisión  según el costo de falsos positivos vs falsos negativos; opcionalmente optimizar con curvas Precision-Recall o con pérdida de negocio.
- **Más modelos y A/B testing:** Seguir probando otros algoritmos o ensembles. en producción, A/B tests para comparar versiones del modelo (v1 vs v2) antes de cortar tráfico.
- **Explicabilidad:** Añadir importancia de variables o explicaciones por transacción (SHAP/LIME) para soporte y cumplimiento.
- **Reentrenamiento y retraining pipeline:** Pipeline programado o disparado por eventos para reentrenar con datos recientes y reexportar a una nueva versión (`models/v2/`).

---

## Estructura del proyecto

```
├── Deteccion de fraude.ipynb   # EDA, preprocesamiento, modelos y exportación
├── creditcard.csv              # Dataset (no incluido en el repo; ver Dataset)
├── src/
│   └── preprocessing.py        # Pipeline: log1p(Amount) + RobustScaler
├── scripts/
│   └── export_model.py         # export_model() y load_model(version)
├── models/
│   └── v1/
│       ├── model.joblib
│       ├── preprocessor.joblib
│       └── metadata.json
├── api/
│   ├── main.py                 # FastAPI: GET /health, POST /predict
│   ├── schemas.py              # Pydantic (TransactionInput, PredictResponse)
│   └── requirements.txt
├── Dockerfile
├── .dockerignore
└── PLAN_PRODUCCION.md
```

---

## Requisitos

- Python 3.10+
- Notebook: pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, xgboost, lightgbm
- API: ver `api/requirements.txt`

---

## Dataset

El archivo **creditcard.csv** no se incluye en el repositorio. 

1. Descargalo desde [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Colocalo en la raíz del proyecto.

Columnas: `Time`, `V1`–`V28` (PCA anonimizado), `Amount`, `Class` (0 = normal, 1 = fraude).

---

## Cómo ejecutar y usar

### Entrenamiento y exportación

1. Abrí **Deteccion de fraude.ipynb** y ejecutá las celdas en orden.
2. Al final del flujo se exporta el modelo a `models/v1/`.

### API en local

Desde la raíz del proyecto:

```bash
pip install -r api/requirements.txt
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- **Health:** http://localhost:8000/health  
- **Docs:** http://localhost:8000/docs  
- **Predict:** `POST http://localhost:8000/predict` con JSON (Time, V1–V28, Amount).

### API con Docker

```bash
docker build -t fraude-api .
docker run -p 8000:8000 fraude-api
```

---

## Ejemplo de request y response (`/predict`)

**Request (fragmento):**

```json
{
  "Time": 0,
  "V1": -1.36,
  "V2": -0.07,
  "V3": 1.72,
  "V4": -0.49,
  "V5": -0.23,
  "V6": 0.28,
  "V7": 0.79,
  "V8": 0.17,
  "V9": -0.15,
  "V10": -0.22,
  "V11": -0.18,
  "V12": -0.17,
  "V13": -0.09,
  "V14": -0.43,
  "V15": -0.01,
  "V16": -0.14,
  "V17": 0.07,
  "V18": 0.02,
  "V19": -0.06,
  "V20": 0.01,
  "V21": -0.02,
  "V22": 0.03,
  "V23": 0.02,
  "V24": -0.01,
  "V25": 0.01,
  "V26": 0.0,
  "V27": 0.01,
  "V28": 0.0,
  "Amount": 10.5
}
```

**Response ejemplo:**

```json
{
  "is_fraud": false,
  "score": 0.023,
  "version": "v1"
}
```

---

## Licencia y atribución

Uso educativo / portfolio. El dataset pertenece a sus autores (Kaggle/ULB).
