.DEFAULT_GOAL := help
SHELL := /bin/bash
COMPOSE := docker compose

# ── k3d / Kubernetes config ──────────────────────────────────────
K3D_CLUSTER   := ml-system
K8S_NAMESPACE := ml-system
HELM_RELEASE  := ml-system
HELM_CHART    := helm/ml-system

# Images built by compose — these names are what `docker compose build` produces.
# k3d needs the exact image name:tag to import into its internal registry.
IMG_SERVING  := ml-system-serving:latest
IMG_TRAINING := ml-system-training:latest
IMG_MLFLOW   := ml-system-mlflow:latest

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
	@grep -E '^[a-zA-Z_.-]+:.*##' $(MAKEFILE_LIST) | \
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
#
# Workflow:
#   1. make k3d.create     ← one-time cluster setup
#   2. make k3d.build      ← build custom images via compose
#   3. make k3d.import     ← push custom images into k3s's containerd
#   4. make k3d.deploy     ← helm install/upgrade
#   5. make k3d.status     ← verify pods are running
#   6. make serve.test     ← smoke test
#
# When you change code:
#   make k3d.build && make k3d.import && make k3d.deploy
# Or simply:
#   make k3d.redeploy
#
# ═══════════════════════════════════════════════════════════════

.PHONY: k3d.create k3d.delete k3d.build k3d.import k3d.deploy k3d.status k3d.logs k3d.shell k3d.redeploy k3d.train k3d.serve.restart k3d.keda.install

k3d.keda.install: ## Install KEDA into the cluster (run once after k3d.create)
	@echo "$(CYAN)Installing KEDA...$(RESET)"
	helm repo add kedacore https://kedacore.github.io/charts 2>/dev/null || true
	helm repo update kedacore
	helm upgrade --install keda kedacore/keda \
		--namespace keda \
		--create-namespace \
		--wait
	@echo "$(GREEN)KEDA installed. Run 'make k3d.deploy' to apply the ScaledObject.$(RESET)"

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
	@echo ""
	k3d cluster create $(K3D_CLUSTER) \
		--port "8000:30000@server:0" \
		--port "5000:30001@server:0" \
		--port "3000:30002@server:0" \
		--port "9090:30003@server:0" \
		--port "9001:30004@server:0" \
		--port "5432:30005@server:0" \
		--port "9000:30006@server:0" \
		--k3s-arg "--disable=traefik@server:0"
	kubectl create namespace $(K8S_NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -
	kubectl config set-context --current --namespace=$(K8S_NAMESPACE)
	@echo ""
	@echo "$(GREEN)Cluster ready.$(RESET)"
	@echo ""
	@echo "  Next: make k3d.build && make k3d.import && make k3d.deploy"


k3d.delete: ## Delete k3d cluster and all its data
	k3d cluster delete $(K3D_CLUSTER)
	@echo "$(YELLOW)Cluster '$(K3D_CLUSTER)' deleted.$(RESET)"

k3d.build: build ## Build all custom Docker images (serving, mlflow, training)
	@echo "$(GREEN)Images built. Run 'make k3d.import' to load them into the cluster.$(RESET)"

k3d.import: ## Import custom images into k3d's container runtime
	@echo "$(CYAN)Importing images into k3d...$(RESET)"
	@echo ""
	@echo "  Note: only custom-built images need importing."
	@echo "  Public images (postgres, minio, grafana, etc.) are pulled by k3s directly."
	@echo ""
	k3d image import $(IMG_SERVING) -c $(K3D_CLUSTER)
	k3d image import $(IMG_MLFLOW) -c $(K3D_CLUSTER)
	@echo "$(GREEN)Images imported.$(RESET)"

k3d.deploy: ## Deploy (or upgrade) all services with Helm
	@echo "$(CYAN)Deploying with Helm...$(RESET)"
	helm upgrade --install $(HELM_RELEASE) $(HELM_CHART) \
		-f $(HELM_CHART)/values-local.yaml \
		-n $(K8S_NAMESPACE)
	@echo ""
	@echo "$(GREEN)Deployed. Waiting for pods...$(RESET)"
	kubectl wait --for=condition=available deployment --all \
		-n $(K8S_NAMESPACE) --timeout=180s 2>/dev/null || true
	@echo ""
	@$(MAKE) --no-print-directory k3d.status

k3d.status: ## Show pods, services, and endpoints
	@echo "$(CYAN)Pods:$(RESET)"
	@kubectl get pods -n $(K8S_NAMESPACE) -o wide
	@echo ""
	@echo "$(CYAN)Services:$(RESET)"
	@kubectl get svc -n $(K8S_NAMESPACE)
	@echo ""
	@echo "$(CYAN)Access:$(RESET)"
	@echo "  Serving:       http://localhost:8000"
	@echo "  MLflow:        http://localhost:5000"
	@echo "  Grafana:       http://localhost:3000  (admin/admin)"
	@echo "  Prometheus:    http://localhost:9090"
	@echo "  MinIO console: http://localhost:9001  (minioadmin/minioadmin)"
	@echo "  MinIO API:     http://localhost:9000  (minioadmin/minioadmin)"
	@echo "  Postgres:      localhost:5432         (mlflow/mlflow, db=mlflow)"

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

k3d.redeploy: k3d.build k3d.import k3d.deploy ## Rebuild, import, and redeploy (full cycle)
	@# Force rolling restart of deployments that use custom images.
	@# Required because imagePullPolicy:Never + latest tag means k8s won't
	@# detect that the image content changed after k3d image import.
	kubectl rollout restart deployment/fastapi-serving deployment/mlflow -n $(K8S_NAMESPACE)
	kubectl rollout status deployment/fastapi-serving deployment/mlflow -n $(K8S_NAMESPACE) --timeout=120s
	@echo "$(GREEN)Redeploy complete.$(RESET)"

k3d.train: ## Build training image, run training job in k3d, stream logs, clean up
	@echo "$(CYAN)Building training image...$(RESET)"
	$(COMPOSE) build training
	@echo "$(CYAN)Importing training image into k3d...$(RESET)"
	k3d image import $(IMG_TRAINING) -c $(K3D_CLUSTER)
	@# Delete any pod left over from a previous run.
	kubectl delete pod training -n $(K8S_NAMESPACE) --ignore-not-found=true
	@echo "$(CYAN)Starting training pod...$(RESET)"
	kubectl run training \
		--image=$(IMG_TRAINING) \
		--restart=Never \
		--image-pull-policy=Never \
		--env="MLFLOW_TRACKING_URI=http://mlflow:5000" \
		--env="MLFLOW_S3_ENDPOINT_URL=http://minio:9000" \
		--env="AWS_ACCESS_KEY_ID=minioadmin" \
		--env="AWS_SECRET_ACCESS_KEY=minioadmin" \
		--env="DATA_CONTROLLER_DB_URL=postgresql://mlflow:mlflow@postgres:5432/mlflow" \
		--env="DATASET_S3_ENDPOINT_URL=http://minio:9000" \
		--env="DATASET_BUCKET=mnist-dataset" \
		--env="MODEL_NAME=ml_system_model" \
		--env="TRAINING_MAX_EPOCHS=20" \
		--env="TRAINING_SEED=42" \
		--env="TRAINING_BATCH_SIZE=256" \
		--env="TRAINING_LR=1e-3" \
		-n $(K8S_NAMESPACE)
	@echo "$(CYAN)Waiting for pod to start...$(RESET)"
	kubectl wait --for=condition=Ready pod/training -n $(K8S_NAMESPACE) --timeout=120s
	@echo "$(CYAN)Streaming logs (Ctrl+C detaches but pod keeps running):$(RESET)"
	kubectl logs -f training -n $(K8S_NAMESPACE)
	kubectl delete pod training -n $(K8S_NAMESPACE) --ignore-not-found=true
	@echo "$(GREEN)Training complete. MLflow: http://localhost:5000$(RESET)"

k3d.serve.restart: ## Restart serving pod to immediately load the latest model from MLflow
	kubectl rollout restart deployment/fastapi-serving -n $(K8S_NAMESPACE)
	kubectl rollout status deployment/fastapi-serving -n $(K8S_NAMESPACE) --timeout=120s

# ═══════════════════════════════════════════════════════════════
# BUILD
# ═══════════════════════════════════════════════════════════════

.PHONY: build build.serving build.training build.mlflow

build: ## Build all custom images (serving, mlflow, training)
	$(COMPOSE) build serving mlflow training

build.serving: ## Build serving image
	$(COMPOSE) build serving

build.training: ## Build training image
	$(COMPOSE) build training

build.mlflow: ## Build mlflow image
	$(COMPOSE) build mlflow

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

_LOCAL_DB  := postgresql://mlflow:mlflow@localhost:5432/mlflow
_LOCAL_S3  := http://localhost:9000
_DATASET   := mnist-dataset
_MINIO_KEY := minioadmin
_DATA_ENV  := DATA_CONTROLLER_DB_URL=$(_LOCAL_DB) DATASET_S3_ENDPOINT_URL=$(_LOCAL_S3) DATASET_BUCKET=$(_DATASET) AWS_ACCESS_KEY_ID=$(_MINIO_KEY) AWS_SECRET_ACCESS_KEY=$(_MINIO_KEY)

.PHONY: data.prepare data.seed data.verify data.setup data.inspect.training

data.prepare: ## Download MNIST, resize 14x14, partition 10%/90% into data/
	PYTHONPATH=. python scripts/prepare_mnist.py

data.seed: ## Seed v0 dataset into k3d Postgres + MinIO (requires k3d running)
	$(_DATA_ENV) PYTHONPATH=. python scripts/seed_dataset.py

data.verify: ## Verify dataset counts in Postgres and pixel round-trip with MinIO
	$(_DATA_ENV) PYTHONPATH=. python scripts/verify_dataset.py

data.setup: data.prepare data.seed data.verify ## Full pipeline: prepare → seed → verify

data.inspect.training: ## Plot 4x4 grid of random training images with labels
	$(_DATA_ENV) PYTHONPATH=. python scripts/inspect_dataset.py --split train

# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

.PHONY: train.local

train.local: ## Train MNIST model locally against k3d Postgres, MinIO, and MLflow
	$(_DATA_ENV) MLFLOW_TRACKING_URI=http://localhost:5000 PYTHONPATH=. python -m training.main

# ═══════════════════════════════════════════════════════════════
# TESTING + DEBUG
# ═══════════════════════════════════════════════════════════════

.PHONY: test test.unit lint lint.fix format serve.test serve.test.load mlflow.debug mlflow.ui minio.ui clean.pyc

test: test.unit ## Run all tests

test.unit: ## Run unit tests
	PYTHONPATH=. python -m pytest tests/unit/ -v

lint: ## Check code with ruff (no changes)
	ruff check .

lint.fix: ## Auto-fix ruff lint issues and format code
	ruff check --fix .
	ruff format .

format: ## Format code with ruff
	ruff format .

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

mlflow.debug: ## Inspect MLflow artifacts and test ONNX loading
	PYTHONPATH=. python debug_mlflow.py

mlflow.ui: ## Open MLflow UI
	@open http://localhost:5000 2>/dev/null || echo "http://localhost:5000"

minio.ui: ## Open MinIO console
	@open http://localhost:9001 2>/dev/null || echo "http://localhost:9001"

clean.pyc: ## Remove Python cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

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
#   → To add ports: make k3d.delete && make k3d.create && make k3d.redeploy
#
# Start fresh:
#   make k3d.delete && make k3d.create && make k3d.redeploy
#
# ═══════════════════════════════════════════════════════════════


# ╔═══════════════════════════════════════════════════════════════╗
# ║  DEPRECATED — DOCKER COMPOSE                                  ║
# ║                                                               ║
# ║  These targets drove the initial development phase.           ║
# ║  All services now run in Kubernetes (k3d). Use the k3d.*      ║
# ║  targets above instead.                                       ║
# ║                                                               ║
# ║  Kept for reference and emergency local debugging only.       ║
# ╚═══════════════════════════════════════════════════════════════╝

.PHONY: dc.infra.up dc.infra.down dc.infra.logs dc.infra.ps dc.infra.clean
.PHONY: dc.serve dc.serve.down dc.serve.logs dc.up dc.down dc.train

dc.infra.up: ## [DEPRECATED] Start full stack in Docker Compose
	@echo "$(YELLOW)[DEPRECATED] Use 'make k3d.deploy' instead.$(RESET)"
	@docker network create ml-system_default 2>/dev/null || true
	$(COMPOSE) up -d postgres minio minio-init mlflow prometheus alloy grafana
	@echo "$(GREEN)Waiting for MLflow...$(RESET)"
	@until curl -sf http://localhost:5000/health > /dev/null 2>&1; do sleep 2; done
	@echo "$(GREEN)Waiting for Grafana...$(RESET)"
	@until curl -sf http://localhost:3000/api/health > /dev/null 2>&1; do sleep 2; done
	@echo "$(GREEN)Ready.$(RESET)"

dc.infra.down: ## [DEPRECATED] Stop Docker Compose infrastructure
	@echo "$(YELLOW)[DEPRECATED]$(RESET)"
	$(COMPOSE) down

dc.infra.logs: ## [DEPRECATED] Tail Docker Compose infrastructure logs
	@echo "$(YELLOW)[DEPRECATED]$(RESET)"
	$(COMPOSE) logs -f postgres minio mlflow alloy prometheus grafana

dc.infra.ps: ## [DEPRECATED] Show running Docker Compose services
	@echo "$(YELLOW)[DEPRECATED]$(RESET)"
	$(COMPOSE) ps

dc.infra.clean: ## [DEPRECATED] Stop Docker Compose and destroy all volumes
	@echo "$(YELLOW)[DEPRECATED]$(RESET)"
	$(COMPOSE) down -v

dc.serve: ## [DEPRECATED] Start serving in Docker Compose
	@echo "$(YELLOW)[DEPRECATED] Use 'make k3d.deploy' instead.$(RESET)"
	$(COMPOSE) up -d serving

dc.serve.down: ## [DEPRECATED] Stop serving in Docker Compose
	@echo "$(YELLOW)[DEPRECATED]$(RESET)"
	$(COMPOSE) stop serving

dc.serve.logs: ## [DEPRECATED] Tail serving logs in Docker Compose
	@echo "$(YELLOW)[DEPRECATED]$(RESET)"
	$(COMPOSE) logs -f serving

dc.up: ## [DEPRECATED] Full Docker Compose pipeline (infra + train + serve)
	@echo "$(YELLOW)[DEPRECATED] Use 'make k3d.redeploy' instead.$(RESET)"
	@$(MAKE) dc.infra.up

dc.down: ## [DEPRECATED] Stop all Docker Compose services
	@echo "$(YELLOW)[DEPRECATED]$(RESET)"
	$(COMPOSE) down

dc.train: ## [DEPRECATED] Train in Docker Compose (use train.local instead)
	@echo "$(YELLOW)[DEPRECATED] Use 'make train.local' instead.$(RESET)"
	$(COMPOSE) run --rm training
