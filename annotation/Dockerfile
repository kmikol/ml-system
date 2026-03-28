FROM python:3.11-slim-bookworm
WORKDIR /app
COPY annotation/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ shared/
COPY sampling/ sampling/
COPY annotation/ annotation/
# Oracle files: UUID→label mapping used by the annotation job to look up
# ground truth labels without a database round-trip to a samples table.
COPY data/remaining/uuids.npy data/remaining/uuids.npy
COPY data/remaining/labels.npy data/remaining/labels.npy
CMD ["python", "-m", "annotation.main"]
