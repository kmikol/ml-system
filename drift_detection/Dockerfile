FROM python:3.11-slim-bookworm
WORKDIR /app
COPY drift_detection/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ shared/
COPY drift_detection/ drift_detection/
CMD ["python", "-m", "drift_detection.main"]
