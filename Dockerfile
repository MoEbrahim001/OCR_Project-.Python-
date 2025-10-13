FROM python:3.12-slim

# --- OS deps (Tesseract + OpenCV + SQL Server ODBC) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg2 ca-certificates apt-transport-https \
    unixodbc unixodbc-dev \
    tesseract-ocr \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

# Microsoft repo for ODBC driver (Debian 12 / bookworm)
RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/microsoft.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" > /etc/apt/sources.list.d/microsoft-prod.list \
 && apt-get update \
 && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 mssql-tools18 \
 && rm -rf /var/lib/apt/lists/*

ENV PATH="${PATH}:/opt/mssql-tools18/bin"

# --- Tesseract language data (copy your custom files) ---
# Make sure these files exist locally at ./tessdata/ (next to Dockerfile)
COPY tessdata/ /usr/share/tesseract-ocr/5/tessdata/
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

# --- Python deps ---
# IMPORTANT: python-multipart is required for FastAPI UploadFile/form-data
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- App code ---
WORKDIR /app
COPY . /app

# --- Run FastAPI ---
EXPOSE 8000
CMD ["uvicorn", "RestAPI:app", "--host", "0.0.0.0", "--port", "8000"]

