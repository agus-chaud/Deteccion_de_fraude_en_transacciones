# Imagen base: Python 3.11 en Linux slim
FROM python:3.11-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar dependencias de la API
COPY api/requirements.txt requirements.txt

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código y artefactos necesarios para la API
# (main.py importa scripts.export_model y load_model busca models/)
COPY api/ api/
COPY models/ models/
COPY scripts/ scripts/
COPY src/ src/

# La API escucha en el puerto 8000
EXPOSE 8000

# Comando por defecto: levantar la API
# api.main:app = módulo api.main, variable app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
