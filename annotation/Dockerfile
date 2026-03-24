FROM python:3.11-slim-bookworm
WORKDIR /app
COPY annotation/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ shared/
COPY annotation/ annotation/
CMD ["python", "-m", "annotation.main"]
