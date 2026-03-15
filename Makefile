.DEFAULT_GOAL := help
SHELL := /bin/bash
COMPOSE := docker compose

CYAN  := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RESET := \033[0m

.PHONY: help
help: ## Show this help
	@echo ""
	@echo "$(CYAN)ML System$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_.-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  $(GREEN)%-22s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ═══════════════════════════════════════════════════════════════
# INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════

.PHONY: infra.up infra.down infra.logs infra.ps infra.clean

infra.up: ## Start infrastructure (postgres, minio, mlflow)
	@docker network create ml-system_default 2>/dev/null || true
	$(COMPOSE) up -d postgres minio minio-init mlflow
	@echo "$(GREEN)Waiting for MLflow...$(RESET)"
	@until curl -sf http://localhost:5000/health > /dev/null 2>&1; do sleep 2; done
	@echo "$(GREEN)Ready.$(RESET)"
	@echo "  MLflow:  http://localhost:5000"
	@echo "  MinIO:   http://localhost:9001"

infra.down: ## Stop infrastructure
	$(COMPOSE) down

infra.logs: ## Tail infrastructure logs
	$(COMPOSE) logs -f postgres minio mlflow

infra.ps: ## Show running services
	$(COMPOSE) ps

infra.clean: ## Stop and destroy all volumes
	$(COMPOSE) down -v
	@echo "$(YELLOW)All volumes removed.$(RESET)"

# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

.PHONY: train train.local

train: ## Train model in Docker
	$(COMPOSE) run --rm training
	@echo "$(GREEN)Training complete. Check MLflow: http://localhost:5000$(RESET)"

train.local: ## Train model locally (devcontainer)
	PYTHONPATH=. python -m training.main

# ═══════════════════════════════════════════════════════════════
# SERVING
# ═══════════════════════════════════════════════════════════════

.PHONY: serve serve.local serve.down serve.logs serve.test

serve: ## Start serving container
	$(COMPOSE) up -d serving
	@echo "$(GREEN)Waiting for serving...$(RESET)"
	@until curl -sf http://localhost:8000/health > /dev/null 2>&1; do sleep 2; done
	@echo "$(GREEN)Serving ready: http://localhost:8000$(RESET)"

serve.local: ## Run serving locally (devcontainer)
	PYTHONPATH=. uvicorn serving.main:app --host 0.0.0.0 --port 8000 --reload

serve.down: ## Stop serving
	$(COMPOSE) stop serving

serve.logs: ## Tail serving logs
	$(COMPOSE) logs -f serving

serve.test: ## Smoke test against running serving
	@echo "$(CYAN)Health:$(RESET)"
	@curl -s http://localhost:8000/health | python3 -m json.tool
	@echo "\n$(CYAN)Predict:$(RESET)"
	@curl -s -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d '{"features": {"age": 35, "income": 55000, "credit_score": 720, "debt_ratio": 1.2, "num_accounts": 5}}' \
		| python3 -m json.tool

# ═══════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════

.PHONY: up down

up: infra.up train serve ## Full Phase 1: infra → train → serve
	@echo "$(GREEN)Phase 1 running.$(RESET)"

down: ## Stop everything
	$(COMPOSE) down

# ═══════════════════════════════════════════════════════════════
# BUILD
# ═══════════════════════════════════════════════════════════════

.PHONY: build build.serving build.training build.mlflow

build: ## Build all images
	$(COMPOSE) build

build.serving: ## Build serving image
	$(COMPOSE) build serving

build.training: ## Build training image
	$(COMPOSE) build training

build.mlflow: ## Build mlflow image
	$(COMPOSE) build mlflow

# ═══════════════════════════════════════════════════════════════
# TESTING + DEBUG
# ═══════════════════════════════════════════════════════════════

.PHONY: test test.unit lint mlflow.debug mlflow.ui minio.ui clean.pyc

test: test.unit ## Run all tests

test.unit: ## Run unit tests
	PYTHONPATH=. python -m pytest tests/unit/ -v

lint: ## Lint
	ruff check .

mlflow.debug: ## Inspect MLflow artifacts and test ONNX loading
	PYTHONPATH=. python debug_mlflow.py

mlflow.ui: ## Open MLflow UI
	@open http://localhost:5000 2>/dev/null || echo "http://localhost:5000"

minio.ui: ## Open MinIO console
	@open http://localhost:9001 2>/dev/null || echo "http://localhost:9001"

clean.pyc: ## Remove Python cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ═══════════════════════════════════════════════════════════════
# KUBERNETES (k3d)
# ═══════════════════════════════════════════════════════════════

.PHONY: k3d.create k3d.delete k3d.load k3d.deploy k3d.status

k3d.create: ## Create k3d cluster
	k3d cluster create ml-system \
		--port "8000:30000@server:0" \
		--port "5000:30001@server:0" \
		--port "3000:30002@server:0" \
		--port "9090:30003@server:0" \
		--k3s-arg "--disable=traefik@server:0"
	kubectl create namespace ml-system --dry-run=client -o yaml | kubectl apply -f -
	kubectl config set-context --current --namespace=ml-system
	@echo "$(GREEN)Cluster ready.$(RESET)"

k3d.delete: ## Delete k3d cluster
	k3d cluster delete ml-system

k3d.load: build ## Load images into k3d
	k3d image import ml-system-serving:latest ml-system-training:latest -c ml-system

k3d.deploy: ## Deploy with Helm
	helm upgrade --install ml-system helm/ml-system/ -f helm/ml-system/values-local.yaml -n ml-system

k3d.status: ## Cluster status
	@kubectl get pods,svc -n ml-system
