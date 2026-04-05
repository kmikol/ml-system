# Values & Configuration

Documentation automatically syncs with Helm configuration — no manual updates needed.

## How It Works

**Template placeholders** in documentation get replaced with actual Helm values:

```markdown
# Template (docs/templates/index.md)
- PSI threshold: {{ .Values.alerts.drift.psiThreshold }}

# Output (docs/build/index.md)
- PSI threshold: 0.25
```

## Quick Start

**1. Change a value in Helm:**
```bash
vim helm/ml-system/values.yaml
```

**2. Rebuild docs:**
```bash
make docs.populate
```

**3. Docs auto-update** with new values.

## Placeholder Format

Documentation uses: `{{ .Values.path.to.key }}`

Supports:
- Nested objects: `{{ .Values.alerts.drift.psiThreshold }}`
- Array access: `{{ .Values.rollout.canarySteps[0].setWeight }}`

## Files Updated

- `docs/build/index.md` — Key thresholds and parameters
- `docs/build/end-to-end-flow.md` — Alert conditions, canary progression
- `docs/build/architecture.md` — Design details, thresholds

## Current Values

From `helm/ml-system/values.yaml`:

| Setting | Value | Purpose |
|---------|-------|---------|
| PSI threshold | 0.25 | Drift alert trigger |
| Annotation count | 50 | Retraining readiness |
| Latency p99 | 1.0s | SLA for serving |
| Canary phase 1 | 20% for 2m | Initial traffic shift |
| Canary phase 2 | 40% for 2m | Increase traffic |

## Adding New Placeholders

1. Add value to `helm/ml-system/values.yaml`
2. Use placeholder in template: `{{ .Values.new.value }}`
3. Run `make docs.populate`

## Preview Changes

Check what would be updated without modifying files:

```bash
make docs.populate.check
```

## CI/CD Integration

Add to pipeline to ensure docs stay in sync:

```bash
make docs.populate
git diff --exit-code docs/build/
```
