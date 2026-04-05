# Running

This section covers the system execution path from bootstrap to traffic simulation.

## Bootstrap Cluster and Workflows

```bash
make bootstrap
```

`make bootstrap` executes the full startup path via `k3d.bootstrap`:

- create k3d cluster
- install KEDA and Argo components
- build and import images
- deploy Helm resources
- run bootstrap workflow (seed -> verify -> train -> restart serving)

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

Example:

```bash
RATE=8 DURATION=120 INVERSION_PROB=0.7 make test.serve.drift
```
