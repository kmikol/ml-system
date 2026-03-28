FROM python:3.11-slim-bookworm
WORKDIR /app
COPY monitoring/ml_exporter/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ shared/
COPY monitoring/ml_exporter/ monitoring/ml_exporter/
EXPOSE 8001
CMD ["uvicorn", "monitoring.ml_exporter.main:app", "--host", "0.0.0.0", "--port", "8001"]
