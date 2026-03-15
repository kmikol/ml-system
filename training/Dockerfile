FROM python:3.11-slim-bookworm
WORKDIR /app
COPY training/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ shared/
COPY training/ training/
CMD ["python", "-m", "training.main"]
