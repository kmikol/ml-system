# Running

This section covers the system execution path from initialize to traffic simulation.

## Minimum Run (Local Setup Only)

If you just want the shortest path to run the system locally for testing:

```bash
make initialize
make test.serve
```

This path will:

- initialize cluster and workloads
- run seed/verify/train/restart workflow
- verify serving endpoint responds locally

## Bootstrap Cluster and Workflows

```bash
make initialize
```

`make initialize` executes the full startup path via `k3d.initialize`:

- create k3d cluster
- install KEDA and Argo components
- build and import images
- deploy Helm resources
- run initialization workflow (seed -> verify -> train -> restart serving)

## Important Make Targets

Core targets used most often during local testing:

- `make initialize` — full first-time startup path
- `make k3d.status` — current cluster and service state
- `make test.serve` — smoke test inference endpoint
- `make test.serve.load` — load test serving
- `make test.serve.drift` — data-drift simulation that should trigger retraining flow
- `make k3d.redeploy` — rebuild/import/redeploy after code changes
- `make k3d.logs POD=fastapi-serving` — tail serving logs

## Verify Readiness

```bash
make k3d.status
make test.serve
```

Expected outcome:

- pods are running in namespaces `ml-system`, `argo`, and `argo-events`
- serving health and prediction endpoint respond successfully

## Run Load and Drift Tests

Load test:

```bash
make test.serve.load
```

Example:

```bash
RATE=10 DURATION=90 make test.serve.load
```

Drift test:

```bash
make test.serve.drift
```

What this test does:

- simulates data drift by inverting image from black digits on white background to white digits on black background
- sends drifted traffic to serving, which should shift feature/prediction distributions
- is intended to push drift metrics high enough that PSI threshold is exceeded

Expected system behavior after sustained drift traffic:

- PSI threshold is exceeded in monitoring
- alert/event path triggers annotation workflow
- new labels are integrated and retraining workflow is triggered
- a new model version is produced and rolled out

Example:

```bash
RATE=10 DURATION=1200 INVERSION_PROB=0.7 make test.serve.drift
```

## Local UIs (Works With Local Setup Only)

These links assume local k3d port mappings from this project setup:

- Grafana: <http://localhost/grafana>
- Prometheus: <http://localhost/prometheus>
- MLflow: <http://localhost/mlflow>
- Argo Workflows: <http://localhost:2746>

