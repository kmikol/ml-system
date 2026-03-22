FROM python:3.11-slim-bookworm

COPY tests/requirements.txt /tmp/tests-req.txt
RUN pip install --no-cache-dir -r /tmp/tests-req.txt

ENV PYTHONPATH=/workspace
WORKDIR /workspace
