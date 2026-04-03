.DEFAULT_GOAL := help
SHELL := /bin/bash

# ── Project config ────────────────────────────────────────────────
# These values must mirror helm/ml-system/values-local.yaml.
# Used by Makefile targets that run one-off pods (k3d.train,
# k3d.annotate) and local scripts (data.*, train.local).
AWS_ACCESS_KEY_ID     := minioadmin
AWS_SECRET_ACCESS_KEY := minioadmin
MLFLOW_S3_ENDPOINT_URL := http://minio:9000
POSTGRES_USER    := mlflow
POSTGRES_PASSWORD := mlflow
POSTGRES_DB      := mlflow
DATA_CONTROLLER_DB_URL := postgresql://mlflow:mlflow@postgres:5432/mlflow
MLFLOW_TRACKING_URI := http://mlflow:5000
MODEL_NAME       := ml_system_model
TRAINING_MAX_EPOCHS := 20
TRAINING_SEED    := 42
TRAINING_BATCH_SIZE := 256
TRAINING_LR      := 1e-3
DATASET_BUCKET   := mnist-dataset

# ── k3d / Kubernetes config ──────────────────────────────────────
K3D_CLUSTER   := ml-system
K8S_NAMESPACE := ml-system
HELM_RELEASE  := ml-system
HELM_CHART    := helm/ml-system

# Images built by compose — these names are what `docker compose build` produces.
# k3d needs the exact image name:tag to import into its internal registry.
IMG_SERVING    := ml-system-serving:latest
IMG_TRAINING   := ml-system-training:latest
IMG_MLFLOW     := ml-system-mlflow:latest
IMG_ANNOTATION := ml-system-annotation:latest
IMG_ML_EXPORTER := ml-system-ml-exporter:latest

TEST_COMPOSE := docker compose -f docker-compose.test.yml

CYAN   := \033[36m
GREEN  := \033[32m
YELLOW := \033[33m
RED    := \033[31m
RESET  := \033[0m

.PHONY: help
help: ## Show this help
	@echo ""
	@echo "$(CYAN)ML System$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z0-9_.-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  $(GREEN)%-22s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ═══════════════════════════════════════════════════════════════
# KUBERNETES (k3d)
#
# Architecture on macOS:
#
#   Your Mac (host)
#     └── Docker Desktop
#           ├── k3d-ml-system-server-0    ← Docker container running k3s
#           │     └── containerd          ← k3s's own container runtime
#           │           ├── serving pod
#           │           ├── mlflow pod
#           │           ├── postgres pod
#           │           ├── minio pod
#           │           ├── prometheus pod
#           │           ├── grafana pod
#           │           └── alloy pod
#           └── k3d-ml-system-serverlb    ← nginx proxy for port mapping
#
# Key concepts:
#
# - k3s is a lightweight Kubernetes distro (single binary, ~70MB RAM).
#   It can't run natively on macOS — it needs Linux.
#
# - k3d wraps k3s inside Docker containers. Each "k3s node" is a
#   Docker container. Inside that container, k3s runs its own
#   container runtime (containerd) which runs your pods.
#   So it's: Docker → k3s container → containerd → your pod.
#
# - Because k3s's containerd is SEPARATE from Docker Desktop's daemon,
#   your locally built Docker images are invisible to k3s.
#   You must explicitly import them with `k3d image import`.
#   This is the #1 gotcha. Every time you rebuild, re-import.
#   (Public images like postgres/grafana are pulled normally — no import needed.)
#
# - Port mapping: k3d creates an nginx load balancer container that
#   forwards host ports to k3s NodePorts. The mapping is set at
#   cluster creation time and cannot be changed later.
#   To add ports, delete and recreate the cluster.
#
# - kubectl context: k3d automatically creates a kubectl context
#   named `k3d-{cluster-name}` and sets it as current.
#
# Port map (host → NodePort → service):
#   localhost:8000  →  30000  →  fastapi-serving
#   localhost:5000  →  30001  →  mlflow
#   localhost:3000  →  30002  →  grafana
#   localhost:9090  →  30003  →  prometheus
#   localhost:9001  →  30004  →  minio console
#   localhost:5432  →  30005  →  postgres
#   localhost:9000  →  30006  →  minio API
#   localhost:2746  →  30007  →  argo-workflows UI
#
# First-time setup (one command):
#   make k3d.bootstrap     ← create cluster, install KEDA, build + import images,
#                            deploy, seed dataset, train model, restart serving
#
# Workflow (step-by-step equivalent):
#   1. make k3d.create          ← one-time cluster setup
#   2. make k3d.keda.install    ← install KEDA (one-time)
#   3. make k3d.build           ← build custom images with docker build
#   4. make k3d.import          ← push custom images into k3s's containerd
#   5. make k3d.deploy          ← helm install/upgrade
#   6. make data.setup          ← prepare + seed + verify dataset
#   7. make k3d.train           ← train and register model
#   8. make k3d.serve.restart   ← reload serving with trained model
#   9. make k3d.status          ← verify pods are running
#  10. make serve.test          ← smoke test
#
# When you change code:
#   make k3d.build && make k3d.import && make k3d.deploy
# Or simply:
#   make k3d.redeploy
#
# ═══════════════════════════════════════════════════════════════

.PHONY: k3d.bootstrap k3d.bootstrap.workflow k3d.create k3d.delete.data k3d.delete.all k3d.build k3d.import k3d.deploy k3d.status k3d.logs k3d.shell k3d.redeploy k3d.train k3d.annotate k3d.serve.restart k3d.keda.install k3d.ml-exporter.restart k3d.argo.install

k3d.keda.install: ## Install KEDA into the cluster (run once after k3d.create)
	@echo "$(CYAN)Installing KEDA...$(RESET)"
	helm repo add kedacore https://kedacore.github.io/charts 2>/dev/null || true
	helm repo update kedacore
	helm upgrade --install keda kedacore/keda \
		--namespace keda \
		--create-namespace \
		--wait
	@echo "$(GREEN)KEDA installed. Run 'make k3d.deploy' to apply the ScaledObject.$(RESET)"

k3d.argo.install: ## Install Argo Workflows + Argo Events into the cluster (run once after k3d.create)
	@echo "$(CYAN)Installing Argo Workflows...$(RESET)"
	helm repo add argo https://argoproj.github.io/argo-helm 2>/dev/null || true
	helm repo update argo
	helm upgrade --install argo-workflows argo/argo-workflows \
		--namespace argo \
		--create-namespace \
		--set "server.authModes={server}" \
		--set server.serviceType=NodePort \
		--set server.serviceNodePort=30007 \
		--wait
	@echo "$(CYAN)Installing Argo Events...$(RESET)"
	helm upgrade --install argo-events argo/argo-events \
		--namespace argo-events \
		--create-namespace \
		--set crds.install=true \
		--wait
	@echo "$(GREEN)Argo Workflows UI: http://localhost:2746$(RESET)"
	@echo "$(GREEN)Argo Workflows + Argo Events installed. Run 'make k3d.deploy' to apply EventBus/EventSource/Sensor.$(RESET)"

k3d.bootstrap: ## First-time setup: create cluster, install KEDA+Argo, build+import images, deploy, run bootstrap workflow
	@echo "$(CYAN)╔══════════════════════════════════════════════════════╗$(RESET)"
	@echo "$(CYAN)║  k3d bootstrap - full first-time startup             ║$(RESET)"
	@echo "$(CYAN)╚══════════════════════════════════════════════════════╝$(RESET)"
	@echo ""
	@echo "$(CYAN)[1/8] Creating k3d cluster...$(RESET)"
	@$(MAKE) --no-print-directory k3d.create
	@echo ""
	@echo "$(CYAN)[2/8] Installing KEDA...$(RESET)"
	@$(MAKE) --no-print-directory k3d.keda.install
	@echo ""
	@echo "$(CYAN)[3/8] Installing Argo Workflows + Argo Events...$(RESET)"
	@$(MAKE) --no-print-directory k3d.argo.install
	@echo ""
	@echo "$(CYAN)[4/8] Preparing dataset (download + partition + assign UUIDs)...$(RESET)"
	@$(MAKE) --no-print-directory data.prepare
	@echo ""
	@echo "$(CYAN)[5/8] Building Docker images (data/v0/ + scripts/ baked into training image after prepare)...$(RESET)"
	@$(MAKE) --no-print-directory k3d.build
	@echo ""
	@echo "$(CYAN)[6/8] Importing images into k3d...$(RESET)"
	@$(MAKE) --no-print-directory k3d.import
	@echo ""
	@echo "$(CYAN)[7/8] Deploying services with Helm...$(RESET)"
	@$(MAKE) --no-print-directory k3d.deploy
	@echo ""
	@echo "$(CYAN)[8/8] Submitting bootstrap-init Argo workflow (seed → verify → train → restart-serving)...$(RESET)"
	@$(MAKE) --no-print-directory k3d.bootstrap.workflow
	@echo ""
	@echo "$(GREEN)╔══════════════════════════════════════════════════════╗$(RESET)"
	@echo "$(GREEN)║  Bootstrap complete!                                 ║$(RESET)"
	@echo "$(GREEN)╚══════════════════════════════════════════════════════╝$(RESET)"
	@echo ""
	@$(MAKE) --no-print-directory k3d.status

k3d.bootstrap.workflow: ## Submit bootstrap-init Argo workflow and wait for completion (seed → verify → train → restart-serving)
	@echo "$(CYAN)Deleting any previous bootstrap-init-run workflow...$(RESET)"
	kubectl delete workflow bootstrap-init-run -n argo --ignore-not-found=true
	@echo "$(CYAN)Submitting bootstrap-init workflow...$(RESET)"
	printf '%s\n' \
		'apiVersion: argoproj.io/v1alpha1' \
		'kind: Workflow' \
		'metadata:' \
		'  name: bootstrap-init-run' \
		'  namespace: argo' \
		'spec:' \
		'  workflowTemplateRef:' \
		'    name: bootstrap-init' \
		| kubectl create -f -
	@echo "$(CYAN)Workflow submitted. Monitor at: http://localhost:2746$(RESET)"
	@echo "$(CYAN)Waiting for workflow to complete...$(RESET)"
	@until PHASE=$$(kubectl get workflow bootstrap-init-run -n argo \
		-o jsonpath='{.status.phase}' 2>/dev/null); \
		[ "$$PHASE" = "Succeeded" ] || [ "$$PHASE" = "Failed" ] || [ "$$PHASE" = "Error" ]; do \
		echo "  status: $${PHASE:-Pending}..."; sleep 10; \
	done; \
	PHASE=$$(kubectl get workflow bootstrap-init-run -n argo -o jsonpath='{.status.phase}'); \
	if [ "$$PHASE" != "Succeeded" ]; then \
		echo "$(RED)Bootstrap workflow $$PHASE. Check Argo UI: http://localhost:2746$(RESET)"; \
		exit 1; \
	fi
	@echo "$(GREEN)Bootstrap workflow succeeded.$(RESET)"

k3d.create: ## Create k3d cluster with port mappings
	@echo "$(CYAN)Creating k3d cluster '$(K3D_CLUSTER)'...$(RESET)"
	@echo ""
	@echo "  Port mapping (set at creation, immutable after):"
	@echo "    localhost:8000  →  NodePort 30000  (serving)"
	@echo "    localhost:5000  →  NodePort 30001  (mlflow)"
	@echo "    localhost:3000  →  NodePort 30002  (grafana)"
	@echo "    localhost:9090  →  NodePort 30003  (prometheus)"
	@echo "    localhost:9001  →  NodePort 30004  (minio console)"
	@echo "    localhost:5432  →  NodePort 30005  (postgres)"
	@echo "    localhost:9000  →  NodePort 30006  (minio API)"
	@echo "    localhost:2746  →  NodePort 30007  (argo-workflows UI)"
	@echo ""
	k3d cluster create $(K3D_CLUSTER) \
		--port "8000:30000@server:0" \
		--port "5000:30001@server:0" \
		--port "3000:30002@server:0" \
		--port "9090:30003@server:0" \
		--port "9001:30004@server:0" \
		--port "5432:30005@server:0" \
		--port "9000:30006@server:0" \
		--port "2746:30007@server:0" \
		--k3s-arg "--disable=traefik@server:0"
	kubectl create namespace $(K8S_NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -
	kubectl config set-context --current --namespace=$(K8S_NAMESPACE)
	@echo ""
	@echo "$(GREEN)Cluster ready.$(RESET)"
	@echo ""
	@echo "  First-time setup: make k3d.keda.install && make k3d.build && make k3d.import && make k3d.deploy"
	@echo "  Or run everything at once: make k3d.bootstrap"


k3d.delete.data: ## Purge runtime data (Postgres/MLflow/MinIO/lakeFS PVC+PV) but keep the k3d cluster
	@echo "$(YELLOW)Purging all ml-system runtime data (DB/artifacts/lakeFS objects)...$(RESET)"
	-helm uninstall $(HELM_RELEASE) -n $(K8S_NAMESPACE) --wait
	-kubectl delete pvc -n $(K8S_NAMESPACE) --all --ignore-not-found=true
	@PVS=$$(kubectl get pv -o jsonpath='{range .items[?(@.spec.claimRef.namespace=="$(K8S_NAMESPACE)")]}{.metadata.name}{"\n"}{end}'); \
	if [ -n "$$PVS" ]; then \
		echo "Deleting PVs:" $$PVS; \
		kubectl delete pv $$PVS; \
	fi
	-kubectl create namespace $(K8S_NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -
	@echo "$(GREEN)Data purge complete. Redeploy with: make k3d.deploy$(RESET)"

k3d.delete.all: ## Delete k3d cluster and all its data
	k3d cluster delete $(K3D_CLUSTER)
	@echo "$(YELLOW)Cluster '$(K3D_CLUSTER)' deleted.$(RESET)"

k3d.build: build ## Build all custom Docker images (serving, mlflow, ml-exporter, annotation)
	@echo "$(GREEN)Images built. Run 'make k3d.import' to load them into the cluster.$(RESET)"

k3d.import: ## Import custom images into k3d's container runtime
	@echo "$(CYAN)Importing images into k3d...$(RESET)"
	@echo ""
	@echo "  Note: only custom-built images need importing."
	@echo "  Public images (postgres, minio, grafana, etc.) are pulled by k3s directly."
	@echo ""
	k3d image import $(IMG_SERVING) -c $(K3D_CLUSTER)
	k3d image import $(IMG_MLFLOW) -c $(K3D_CLUSTER)
	k3d image import $(IMG_ML_EXPORTER) -c $(K3D_CLUSTER)
	k3d image import $(IMG_ANNOTATION) -c $(K3D_CLUSTER)
	k3d image import $(IMG_TRAINING) -c $(K3D_CLUSTER)
	@echo "$(GREEN)Images imported.$(RESET)"

k3d.deploy: ## Deploy (or upgrade) all services with Helm, then apply Argo Events CRD instances (fast by default; WAIT=1 for blocking readiness)
	@echo "$(CYAN)Deploying with Helm...$(RESET)"
	helm upgrade --install $(HELM_RELEASE) $(HELM_CHART) \
		-f $(HELM_CHART)/values-local.yaml \
		-n $(K8S_NAMESPACE)
	@echo ""
	@echo "$(CYAN)Applying Argo Events resources (SSA)...$(RESET)"
	kubectl apply --server-side --force-conflicts -f k8s/argo/argo-events-resources.yaml
	kubectl apply --server-side --force-conflicts -f k8s/argo/workflows/
	@echo ""
	@echo "$(GREEN)Deployed.$(RESET)"
	@if [ "$(WAIT)" = "1" ]; then \
		echo "$(CYAN)WAIT=1 set: waiting for deployments to become available...$(RESET)"; \
		kubectl wait --for=condition=available deployment --all \
			-n $(K8S_NAMESPACE) --timeout=$${TIMEOUT:-120s}; \
	else \
		echo "$(YELLOW)Skipping blocking waits (fast mode). Use 'make k3d.deploy WAIT=1' for readiness waits.$(RESET)"; \
	fi

k3d.status: ## Show pods, services, and endpoints
	@echo "$(CYAN)Pods (ml-system):$(RESET)"
	@kubectl get pods -n $(K8S_NAMESPACE) -o wide
	@echo ""
	@echo "$(CYAN)Pods (argo):$(RESET)"
	@kubectl get pods -n argo 2>/dev/null || echo "  (argo namespace not yet installed)"
	@echo ""
	@echo "$(CYAN)Pods (argo-events):$(RESET)"
	@kubectl get pods -n argo-events 2>/dev/null || echo "  (argo-events namespace not yet installed)"
	@echo ""
	@echo "$(CYAN)Services:$(RESET)"
	@kubectl get svc -n $(K8S_NAMESPACE)
	@echo ""
	@echo "$(CYAN)Access:$(RESET)"
	@echo "  Serving:          http://localhost:8000"
	@echo "  MLflow:           http://localhost:5000"
	@echo "  Grafana:          http://localhost:3000  (admin/admin)"
	@echo "  Prometheus:       http://localhost:9090"
	@echo "  Argo Workflows:   http://localhost:2746"
	@echo "  MinIO console:    http://localhost:9001  (minioadmin/minioadmin)"
	@echo "  MinIO API:        http://localhost:9000  (minioadmin/minioadmin)"
	@echo "  Postgres:         localhost:5432         (mlflow/mlflow, db=mlflow)"

k3d.logs: ## Tail logs for a deployment (POD=fastapi-serving)
	@if [ -z "$(POD)" ]; then \
		echo "$(CYAN)Available pods:$(RESET)"; \
		kubectl get pods -n $(K8S_NAMESPACE) --no-headers -o custom-columns=":metadata.name"; \
		echo ""; \
		echo "Usage: make k3d.logs POD=<deployment-name>"; \
	else \
		kubectl logs -f deployment/$(POD) -n $(K8S_NAMESPACE); \
	fi

k3d.shell: ## Open a shell in a pod (POD=fastapi-serving)
	@if [ -z "$(POD)" ]; then \
		echo "$(CYAN)Available pods:$(RESET)"; \
		kubectl get pods -n $(K8S_NAMESPACE) --no-headers -o custom-columns=":metadata.name"; \
		echo ""; \
		echo "Usage: make k3d.shell POD=<deployment-name>"; \
	else \
		kubectl exec -it deployment/$(POD) -n $(K8S_NAMESPACE) -- /bin/bash || \
		kubectl exec -it deployment/$(POD) -n $(K8S_NAMESPACE) -- /bin/sh; \
	fi

k3d.redeploy: k3d.build k3d.import k3d.deploy ## Rebuild, import, and redeploy (fast by default; WAIT=1 to block on rollout)
	@# Force rolling restart of deployments that use custom images.
	@# Required because imagePullPolicy:Never + latest tag means k8s won't
	@# detect that the image content changed after k3d image import.
	kubectl rollout restart deployment/fastapi-serving deployment/mlflow deployment/grafana deployment/alloy deployment/prometheus deployment/alertmanager -n $(K8S_NAMESPACE)
	@kubectl get deployment ml-exporter -n $(K8S_NAMESPACE) &>/dev/null && \
		kubectl rollout restart deployment/ml-exporter -n $(K8S_NAMESPACE) || true
	@if [ "$(WAIT)" = "1" ]; then \
		echo "$(CYAN)WAIT=1 set: waiting for rollout completion...$(RESET)"; \
		kubectl rollout status deployment/fastapi-serving -n $(K8S_NAMESPACE) --timeout=$${TIMEOUT:-120s}; \
		kubectl rollout status deployment/mlflow -n $(K8S_NAMESPACE) --timeout=$${TIMEOUT:-120s}; \
		kubectl rollout status deployment/grafana -n $(K8S_NAMESPACE) --timeout=$${TIMEOUT:-120s}; \
		kubectl rollout status deployment/alloy -n $(K8S_NAMESPACE) --timeout=$${TIMEOUT:-120s}; \
		kubectl rollout status deployment/prometheus -n $(K8S_NAMESPACE) --timeout=$${TIMEOUT:-120s}; \
		kubectl rollout status deployment/alertmanager -n $(K8S_NAMESPACE) --timeout=$${TIMEOUT:-120s}; \
		kubectl get deployment ml-exporter -n $(K8S_NAMESPACE) &>/dev/null && \
			kubectl rollout status deployment/ml-exporter -n $(K8S_NAMESPACE) --timeout=$${TIMEOUT:-120s} || true; \
	else \
		echo "$(YELLOW)Skipping rollout status waits (fast mode). Use 'make k3d.redeploy WAIT=1' to wait.$(RESET)"; \
	fi
	@echo "$(GREEN)Redeploy complete.$(RESET)"

k3d.ml-exporter.restart: ## Restart ml-exporter pod
	kubectl rollout restart deployment/ml-exporter -n $(K8S_NAMESPACE)
	kubectl rollout status deployment/ml-exporter -n $(K8S_NAMESPACE) --timeout=60s

k3d.train: ## Build training image, run training job in k3d, stream logs, clean up
	@echo "$(CYAN)Building training image...$(RESET)"
	docker build -t $(IMG_TRAINING) -f training/Dockerfile .
	@echo "$(CYAN)Importing training image into k3d...$(RESET)"
	k3d image import $(IMG_TRAINING) -c $(K3D_CLUSTER)
	@# Delete any pod left over from a previous run.
	kubectl delete pod training -n $(K8S_NAMESPACE) --ignore-not-found=true
	@echo "$(CYAN)Starting training pod...$(RESET)"
	kubectl run training \
		--image=$(IMG_TRAINING) \
		--restart=Never \
		--image-pull-policy=Never \
		--env="MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI)" \
		--env="MLFLOW_S3_ENDPOINT_URL=$(MLFLOW_S3_ENDPOINT_URL)" \
		--env="AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID)" \
		--env="AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY)" \
		--env="DATA_CONTROLLER_DB_URL=$(DATA_CONTROLLER_DB_URL)" \
		--env="DATASET_S3_ENDPOINT_URL=$(MLFLOW_S3_ENDPOINT_URL)" \
		--env="DATASET_BUCKET=$(DATASET_BUCKET)" \
		--env="MODEL_NAME=$(MODEL_NAME)" \
		--env="TRAINING_MAX_EPOCHS=$(TRAINING_MAX_EPOCHS)" \
		--env="TRAINING_SEED=$(TRAINING_SEED)" \
		--env="TRAINING_BATCH_SIZE=$(TRAINING_BATCH_SIZE)" \
		--env="TRAINING_LR=$(TRAINING_LR)" \
		-n $(K8S_NAMESPACE)
	@echo "$(CYAN)Waiting for pod to start...$(RESET)"
	kubectl wait --for=condition=Ready pod/training -n $(K8S_NAMESPACE) --timeout=120s
	@echo "$(CYAN)Streaming logs (Ctrl+C detaches but pod keeps running):$(RESET)"
	kubectl logs -f training -n $(K8S_NAMESPACE)
	@EXIT_CODE=$$(kubectl get pod training -n $(K8S_NAMESPACE) -o jsonpath='{.status.containerStatuses[0].state.terminated.exitCode}'); \
	if [ -z "$$EXIT_CODE" ]; then EXIT_CODE=1; fi; \
	echo "Training pod exit code: $$EXIT_CODE"; \
	kubectl delete pod training -n $(K8S_NAMESPACE) --ignore-not-found=true; \
	if [ "$$EXIT_CODE" -ne 0 ]; then \
		echo "$(RED)Training failed. See logs above.$(RESET)"; \
		exit $$EXIT_CODE; \
	fi
	@echo "$(GREEN)Training complete. MLflow: http://localhost:5000$(RESET)"

k3d.serve.restart: ## Restart serving pod to immediately load the latest model from MLflow
	kubectl rollout restart deployment/fastapi-serving -n $(K8S_NAMESPACE)
	kubectl rollout status deployment/fastapi-serving -n $(K8S_NAMESPACE) --timeout=120s

k3d.annotate: ## Build annotation image, run annotation job in k3d, stream logs, clean up
	@echo "$(CYAN)Building annotation image...$(RESET)"
	docker build -t $(IMG_ANNOTATION) -f annotation/Dockerfile .
	@echo "$(CYAN)Importing annotation image into k3d...$(RESET)"
	k3d image import $(IMG_ANNOTATION) -c $(K3D_CLUSTER)
	@# Delete any pod left over from a previous run.
	kubectl delete pod annotation -n $(K8S_NAMESPACE) --ignore-not-found=true
	@echo "$(CYAN)Starting annotation pod...$(RESET)"
	kubectl run annotation \
		--image=$(IMG_ANNOTATION) \
		--restart=Never \
		--image-pull-policy=Never \
		--env="DATA_CONTROLLER_DB_URL=$(DATA_CONTROLLER_DB_URL)" \
		--env="ANNOTATION_SAMPLES_PER_RUN=$${ANNOTATION_SAMPLES_PER_RUN:-10}" \
		-n $(K8S_NAMESPACE)
	@echo "$(CYAN)Waiting for pod to start...$(RESET)"
	kubectl wait --for=condition=Ready pod/annotation -n $(K8S_NAMESPACE) --timeout=120s
	@echo "$(CYAN)Streaming logs (Ctrl+C detaches but pod keeps running):$(RESET)"
	kubectl logs -f annotation -n $(K8S_NAMESPACE)
	kubectl delete pod annotation -n $(K8S_NAMESPACE) --ignore-not-found=true
	@echo "$(GREEN)Annotation job complete.$(RESET)"

# ═══════════════════════════════════════════════════════════════
# BUILD
# ═══════════════════════════════════════════════════════════════

.PHONY: build build.serving build.training build.mlflow build.ml-exporter build.annotation

build: build.serving build.mlflow build.ml-exporter build.annotation build.training ## Build all custom images (serving, mlflow, ml-exporter, annotation, training)

build.serving: ## Build serving image
	docker build -t $(IMG_SERVING) -f serving/Dockerfile .

build.training: ## Build training image
	docker build -t $(IMG_TRAINING) -f training/Dockerfile .

build.mlflow: ## Build mlflow image
	docker build -t $(IMG_MLFLOW) -f shared/model_artifact_controller/mlflow/Dockerfile .

build.ml-exporter: ## Build ml-exporter image
	docker build -t $(IMG_ML_EXPORTER) -f monitoring/ml_exporter/Dockerfile .

build.annotation: ## Build annotation image
	docker build -t $(IMG_ANNOTATION) -f annotation/Dockerfile .

# ═══════════════════════════════════════════════════════════════
# DATASET (MNIST)
#
# Requires k3d cluster running (make k3d.redeploy).
# Postgres and MinIO are exposed as NodePorts so local scripts
# connect directly: localhost:5432 (postgres), localhost:9000 (minio API).
#
# Workflow:
#   1. make data.prepare        ← download + resize MNIST locally
#   2. make data.seed           ← upload v0 split to k3d Postgres + MinIO
#   3. make data.verify         ← check counts + pixel round-trip
#
# Or in one shot (after k3d.redeploy): make data.setup
# ═══════════════════════════════════════════════════════════════

# Local-access URLs use localhost NodePorts instead of in-cluster DNS.
_LOCAL_DB  := postgresql://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@localhost:5432/$(POSTGRES_DB)
_LOCAL_S3  := http://localhost:9000
_DATA_ENV  := DATA_CONTROLLER_DB_URL=$(_LOCAL_DB) DATASET_S3_ENDPOINT_URL=$(_LOCAL_S3) DATASET_BUCKET=$(DATASET_BUCKET) AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY)

.PHONY: data.prepare data.seed data.verify data.setup data.inspect.training

data.prepare: ## Download MNIST, resize 14x14, partition 1%/99% into data/
	PYTHONPATH=. python scripts/prepare_mnist.py

data.seed: ## Seed v0 dataset into k3d Postgres + MinIO (requires k3d running)
	$(_DATA_ENV) PYTHONPATH=. python scripts/seed_dataset.py

data.verify: ## Verify dataset counts in Postgres and pixel round-trip with MinIO
	$(_DATA_ENV) PYTHONPATH=. python scripts/verify_dataset.py

data.setup: data.prepare data.seed data.verify ## Full pipeline: prepare → seed → verify

data.inspect.training: ## Plot 4x4 grid of random training images with labels
	$(_DATA_ENV) PYTHONPATH=. python scripts/inspect_dataset.py --split train

# ═══════════════════════════════════════════════════════════════
# TESTING + DEBUG
# ═══════════════════════════════════════════════════════════════

.PHONY: test test.unit test.integration test.helm test.e2e \
        test.data_controller.unit test.data_controller.integration \
        test.model_artifact_controller.unit test.model_artifact_controller.integration \
        lint lint.fix format serve.test serve.test.load serve.test.drift mlflow.ui minio.ui clean.pyc

typecheck: ## Run mypy static type checker
	python -m mypy serving annotation sampling monitoring/ml_exporter shared \
		--ignore-missing-imports --no-strict-optional

lint: ## Check code style with ruff (linting + formatting, no changes)
	ruff check .
	ruff format --check .

lint.fix: ## Auto-fix ruff lint issues and format code
	ruff check --fix .
	ruff format .

format: ## Format code with ruff
	ruff format .

test: ## Run lint, type checking, and all tests (unit + integration + helm) in Docker
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test.helm
	$(TEST_COMPOSE) run --rm test; \
	EXIT=$$?; \
	$(TEST_COMPOSE) down -v; \
	exit $$EXIT

test.unit: ## Run unit tests locally (no Docker needed)
	PYTHONPATH=. python -m pytest tests/unit/ shared/data_controller/tests/unit/ shared/model_artifact_controller/tests/unit/ -v

test.coverage: ## Run unit tests with line-level coverage report (no Docker needed)
	PYTHONPATH=. python -m pytest tests/unit/ shared/data_controller/tests/unit/ shared/model_artifact_controller/tests/unit/ \
		--cov=serving --cov=annotation --cov=sampling --cov=monitoring --cov=shared \
		--cov-report=term-missing \
		-v

test.integration: ## Run integration tests in Docker (data controller + model artifact + serving e2e)
	$(TEST_COMPOSE) run --rm test pytest \
		shared/data_controller/tests/integration/ \
		shared/model_artifact_controller/tests/integration/ \
		tests/integration/test_serving_e2e.py \
		-v; \
	EXIT=$$?; \
	$(TEST_COMPOSE) down -v; \
	exit $$EXIT

test.helm: ## Run Helm chart manifest policy tests locally (requires helm on PATH)
	PYTHONPATH=. python -m pytest tests/helm/ -v --tb=short

test.e2e: ## Run serving e2e tests in Docker (builds serving container, seeds model)
	$(TEST_COMPOSE) run --build --rm test pytest tests/integration/test_serving_e2e.py -v; \
	EXIT=$$?; \
	$(TEST_COMPOSE) down -v; \
	exit $$EXIT

test.data_controller.unit: ## Run data_controller unit tests locally
	PYTHONPATH=. python -m pytest shared/data_controller/tests/unit/ -v

test.data_controller.integration: ## Run data_controller integration tests in Docker
	$(TEST_COMPOSE) run --rm test pytest shared/data_controller/tests/integration/ -v; \
	EXIT=$$?; \
	$(TEST_COMPOSE) down -v; \
	exit $$EXIT

test.model_artifact_controller.unit: ## Run model_artifact_controller unit tests locally
	PYTHONPATH=. python -m pytest shared/model_artifact_controller/tests/unit/ -v

test.model_artifact_controller.integration: ## Run model_artifact_controller integration tests in Docker
	$(TEST_COMPOSE) run --rm test pytest shared/model_artifact_controller/tests/integration/ -v; \
	EXIT=$$?; \
	$(TEST_COMPOSE) down -v; \
	exit $$EXIT


serve.test: ## Smoke test against running serving (works with compose or k3d)
	@echo "$(CYAN)Health:$(RESET)"
	@curl -s http://localhost:8000/health | python3 -m json.tool
	@echo "\n$(CYAN)Predict:$(RESET)"
	@python3 -c "import json; print(json.dumps({'image': [[0.0]*14 for _ in range(14)]}))" \
		| curl -s -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d @- \
		| python3 -m json.tool

serve.test.load: ## Send requests with ramp-up/down (RATE=5 DURATION=60 RAMP_UP=0 RAMP_DOWN=0)
	python3 scripts/load_test.py --rate $${RATE:-5} --duration $${DURATION:-60} \
		--ramp-up $${RAMP_UP:-0} --ramp-down $${RAMP_DOWN:-0}

serve.test.drift: ## Send inverted images with ramping probability (RATE=5 DURATION=120 INVERSION_PROB=1.0 RAMP=60)
	python3 scripts/drift_test.py --rate $${RATE:-5} --duration $${DURATION:-120} \
		--inversion-probability $${INVERSION_PROB:-1.0} --ramp $${RAMP:-60}

clean.pyc: ## Remove Python cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ═══════════════════════════════════════════════════════════════
# DOCS
# ═══════════════════════════════════════════════════════════════

.PHONY: docs.serve.local docs.serve.online

docs.serve.local: ## Serve docs locally with live reload (http://localhost:8001)
	mkdocs serve -f docs/mkdocs.yml --dev-addr 127.0.0.1:8001

docs.serve.online: ## Build and deploy docs to GitHub Pages (gh-pages branch)
	mkdocs gh-deploy -f docs/mkdocs.yml --force

# ═══════════════════════════════════════════════════════════════
# KUBERNETES DEBUGGING
#
# Pod stuck in ImagePullBackOff:
#   → Forgot to import the image: make k3d.import
#   → Or imagePullPolicy is not "Never" in values-local.yaml
#   → Check: kubectl describe pod <name> -n ml-system
#
# Pod stuck in CrashLoopBackOff:
#   → Container starts and immediately crashes.
#   → Check logs: kubectl logs <pod-name> -n ml-system
#   → Usually a missing env var or failed initContainer
#
# Pod is Running but service unreachable:
#   → kubectl get svc -n ml-system
#   → kubectl get endpoints -n ml-system
#   → If endpoints show <none>, selector doesn't match pod labels.
#
# Port not reachable from host:
#   → docker ps | grep k3d  (verify port mapping)
#   → Port mapping is set at cluster creation — immutable.
#   → To add ports: make k3d.delete.all && make k3d.create && make k3d.redeploy
#
# Start fresh:
#   make k3d.delete.all && make k3d.create && make k3d.redeploy
#
# ═══════════════════════════════════════════════════════════════
