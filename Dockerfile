FROM python:3.12-slim

WORKDIR /workspace

COPY src/ ./src/
COPY requirements.txt ./
COPY models/ ./models/
COPY web-metadata/ ./web-metadata/

RUN pip install --no-cache-dir -r requirements.txt