FROM python:3.11-slim-bookworm
WORKDIR /app
COPY monitoring/drift/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ shared/
COPY monitoring/drift/ monitoring/drift/
EXPOSE 8001
CMD ["uvicorn", "monitoring.drift.main:app", "--host", "0.0.0.0", "--port", "8001"]
