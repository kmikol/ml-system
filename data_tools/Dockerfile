FROM python:3.11-slim-bookworm
WORKDIR /app
COPY data_tools/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ shared/
COPY data_tools/ data_tools/
CMD ["python", "-m", "data_tools.export_labels"]
