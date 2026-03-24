# syntax=docker/dockerfile:1.7
FROM python:3.11-slim-bookworm
WORKDIR /app
COPY training/requirements.txt requirements.txt

# Install CPU-only torch first, then install remaining deps using a cached pip directory.
RUN --mount=type=cache,target=/root/.cache/pip \
	pip install --no-cache-dir \
	--index-url https://download.pytorch.org/whl/cpu \
	--extra-index-url https://pypi.org/simple \
	torch==2.2.2
RUN --mount=type=cache,target=/root/.cache/pip \
	pip install --no-cache-dir --extra-index-url https://pypi.org/simple -r requirements.txt
COPY shared/ shared/
COPY training/ training/
CMD ["python", "-m", "training.main"]
