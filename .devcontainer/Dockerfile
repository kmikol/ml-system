FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl vim jq bash-completion make \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/$(dpkg --print-architecture)/kubectl" \
    && chmod +x kubectl && mv kubectl /usr/local/bin/

RUN curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

WORKDIR /workspace

COPY training/requirements.txt /tmp/training-req.txt
COPY serving/requirements.txt /tmp/serving-req.txt
COPY tests/requirements.txt /tmp/tests-req.txt
RUN pip install --no-cache-dir \
    -r /tmp/training-req.txt \
    -r /tmp/serving-req.txt \
    -r /tmp/tests-req.txt \
    pytest ruff mypy httpx ipython

ENV PYTHONPATH=/workspace
