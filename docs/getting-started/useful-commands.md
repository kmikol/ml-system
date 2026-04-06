# Useful Commands

Quick commands for operating and debugging the running system.

## Status and Smoke Test

```bash
make k3d.status
make test.serve
```

## Logs

```bash
make k3d.logs POD=fastapi-serving
make k3d.logs POD=ml-exporter
```

## Redeploy Cycle

```bash
make k3d.redeploy
```

## Train and Restart Serving

```bash
make k3d.train
make k3d.serve.restart
```

## TODO Placeholders

```bash
# TODO: add Prometheus query checklist for load/drift validation
# TODO: add Argo workflow troubleshooting commands for bootstrap failures
```
