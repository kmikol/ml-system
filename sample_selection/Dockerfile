FROM python:3.11-slim-bookworm
WORKDIR /app
COPY sample_selection/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ shared/
COPY sample_selection/ sample_selection/
CMD ["python", "-m", "sample_selection.main"]
