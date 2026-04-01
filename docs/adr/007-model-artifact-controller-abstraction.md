# ADR-007: Model Artifact Controller Boundary

**Date**: 2026-03-30  
**Status**: Proposed  
**Deciders**: ML Platform Team

## Purpose

Define a simple and durable boundary for model artifact and registry operations so training, serving, monitoring, and workflow scripts can use one interface and stay independent from backend details.

This ADR focuses on:

1. The intended implementation idea (simple, elegant, fit-for-purpose)
2. What is implemented today
3. The gap between intended and current state
4. The concrete path to close that gap

## Target Idea

The ModelArtifactController should be a thin boundary with one job: provide the minimum operations the system needs to publish and consume model artifacts safely.

Design principles:

1. One public interface for all services
2. No backend-specific access outside the controller package
3. One error contract for callers
4. Canonical artifact layout managed in one place
5. Alias operations modeled explicitly (not as side effects)
6. Small API surface that maps directly to real use cases

In short: keep service code business-focused, and keep storage/registry mechanics inside the controller.

## Current Implementation

Current implementation already has the right structural shape:

1. Facade in [shared/model_artifact_controller/__init__.py](shared/model_artifact_controller/__init__.py)
2. Protocol contract in [shared/model_artifact_controller/_protocol.py](shared/model_artifact_controller/_protocol.py)
3. MLflow backend in [shared/model_artifact_controller/mlflow.py](shared/model_artifact_controller/mlflow.py)
4. Unit and integration tests in [shared/model_artifact_controller/tests/unit/test_mlflow_controller.py](shared/model_artifact_controller/tests/unit/test_mlflow_controller.py) and [shared/model_artifact_controller/tests/integration/test_mlflow.py](shared/model_artifact_controller/tests/integration/test_mlflow.py)

Current behavior that is correct and useful:

1. Training can start runs, log params/metrics/artifacts, register versions, and promote to Production.
2. Serving can resolve run by alias, download serving bundle, and hot-reload ONNX sessions.
3. Exporter can load reference distribution by alias for drift computation.
4. The backend wraps most failures into ModelArtifactError.

Canonical artifact contract currently used:

- onnx/classifier/model.onnx
- onnx/embedder/model.onnx
- reference_distribution.json
- class_gaussians.json
- feature_schema.json

## Gap Analysis

The architecture idea is mostly right, but implementation discipline is incomplete.

### Gap 1: Boundary is bypassed in scripts

- [scripts/set_alias.py](scripts/set_alias.py) accesses controller internals and calls backend client directly.
- [scripts/evaluate_and_promote.py](scripts/evaluate_and_promote.py) mixes controller calls with direct MLflow client usage.

Impact:

- Backend swap is harder than expected.
- Error handling and behavior are inconsistent.
- Private internals become accidental public API.

### Gap 2: Public API naming and capability mismatch

- Method name get_production_run_id accepts generic alias values.
- Controller does not expose first-class alias management for set and clear flows required by workflows.

Impact:

- Callers work around API intent.
- Naming causes confusion for maintainers.

### Gap 3: Error contract is not fully consistent

- Most backend failures become ModelArtifactError.
- Some paths still raise raw exceptions (for example, missing expected ONNX path).
- Optional artifact loading can silently mask non-optional failure causes.

Impact:

- Caller behavior depends on implementation detail, not interface contract.
- Debugging production issues takes longer.

### Gap 4: Payload contract is weakly typed

- JSON payloads are passed as generic dictionaries without strict boundary validation.

Impact:

- Contract violations fail late in serving or exporter.
- Upgrade safety is lower than needed.

### Gap 5: Duplicate backend operations outside controller

- Version lookup and alias workflows are partly reimplemented in scripts.

Impact:

- Logic duplication and drift risk.
- More brittle workflow behavior.

## Fix Plan

### Phase 1: Close the interface holes

Add the missing operations to the controller contract and backend:

1. get_run_id_by_alias(model_name, alias)
2. set_alias(model_name, alias, version)
3. delete_alias(model_name, alias)
4. get_version_by_run_id(model_name, run_id)

Then migrate scripts to only use controller methods.

Success criteria:

1. No script accesses _backend or _client.
2. No script imports backend-specific client for operations covered by controller.

### Phase 2: Normalize error behavior

1. Ensure all backend-raised exceptions are wrapped as ModelArtifactError at controller boundary.
2. Separate optional-artifact-not-found from malformed-artifact or transport failure.
3. Emit explicit warning messages with enough context for operations.

Success criteria:

1. Callers handle one primary exception type.
2. Optional artifact behavior is explicit and test-covered.

### Phase 3: Harden payload contracts

1. Introduce typed models for reference distribution and class gaussians.
2. Validate payloads at upload and download boundaries.
3. Add schema version field for artifact payload evolution.

Success criteria:

1. Invalid payloads fail early at controller boundary.
2. Serving and exporter no longer rely on implicit dictionary shape.

### Phase 4: Keep the surface small

1. Keep controller API limited to operations used by training, serving, exporter, and workflows.
2. Avoid adding backend-specific convenience methods to facade.
3. Add concise architecture note in module docs describing allowed usage.

Success criteria:

1. API remains purpose-driven and stable.
2. New consumers follow boundary without bypasses.

## Definition of Done

This ADR is considered implemented when:

1. All model registry and artifact operations in application code go through ModelArtifactController.
2. Alias workflows are fully represented in the public contract.
3. Exception behavior is consistent and documented.
4. Payloads are validated at the controller boundary.
5. Tests cover normal flows and failure modes for these contracts.

## Related Decisions

- [ADR-004](004-mlflow-artifact-storage.md): MLflow + PostgreSQL + MinIO backend decision
- [ADR-002](002-canary-rollouts.md): stable/canary alias behavior
- [ADR-005](005-prometheus-monitoring-alerting.md): exporter depends on reference artifacts
