# Model Artifact Controller

The Model Artifact Controller facade encapsulates model artifact logging, registration, and retrieval.

It separates training-side publishing APIs from serving-side retrieval APIs.

## Submodule: Training Logger

Source: `shared/model_artifact_controller/_training_logger.py`

### Facade Routines

::: shared.model_artifact_controller._training_logger.TrainingLogger
		options:
			show_root_heading: true

### Supporting Runtime Handle

::: shared.model_artifact_controller._training_logger.TrainingRun
		options:
			show_root_heading: true

---

## Submodule: Model Store

Source: `shared/model_artifact_controller/_model_store.py`

### Facade Routines

::: shared.model_artifact_controller._model_store.ModelStore
		options:
			show_root_heading: true

---

## Submodule: Public Types

Source: `shared/model_artifact_controller/types.py`

### Enum Types

::: shared.model_artifact_controller.types.ModelStage
		options:
			show_root_heading: true

### Dataclasses

::: shared.model_artifact_controller.types.ReferenceDistribution
		options:
			show_root_heading: true

::: shared.model_artifact_controller.types.ClassGaussian
		options:
			show_root_heading: true

::: shared.model_artifact_controller.types.ClassGaussians
		options:
			show_root_heading: true

::: shared.model_artifact_controller.types.FeatureSchema
		options:
			show_root_heading: true

::: shared.model_artifact_controller.types.TrainingArtifacts
		options:
			show_root_heading: true

::: shared.model_artifact_controller.types.ServingBundle
		options:
			show_root_heading: true

::: shared.model_artifact_controller.types.VersionArtifacts
		options:
			show_root_heading: true

::: shared.model_artifact_controller.types.ModelVersion
		options:
			show_root_heading: true

---

## Submodule: Protocol and Errors

Source: `shared/model_artifact_controller/_protocol.py`

### Error Type

::: shared.model_artifact_controller._protocol.ModelArtifactError
		options:
			show_root_heading: true

### Backend Protocol Routines

::: shared.model_artifact_controller._protocol.ModelArtifactController
		options:
			show_root_heading: true

---

## Submodule: MLflow Backend

Source: `shared/model_artifact_controller/mlflow.py`

### Backend Routines

::: shared.model_artifact_controller.mlflow.MLflowModelArtifactController
		options:
			show_root_heading: true
