# Setup

This project is first and foremost a **local setup for testing the full system**.

Reference machine used for development and verification:

- Apple MacBook Air M2
- 16 GB RAM
- macOS

The system can run on other machines, but setup might differ.

Validated local toolchain versions on this reference machine:

- Docker: 29.2.1
- kubectl: v1.35.2
- k3d: v5.8.3 (k3s v1.33.6-k3s1)
- Helm: v4.1.3
- Conda: 22.11.1
- GNU Make: 3.81

## Why Conda (not a devcontainer)

This project uses a local Conda environment for Python tooling instead of relying on a devcontainer setup.

Reasoning:

- most system components run inside Kubernetes pods anyway
- local development mainly orchestrates cluster actions and runs helper scripts
- adding a full devcontainer here would introduce extra setup overhead with limited payoff

## Prerequisites

- Docker Desktop
- Conda (Miniconda or Anaconda)
- `kubectl`
- `k3d`
- `helm`
- `make`

## 1. Install System Tooling (macOS)

Install Xcode command line tools:

```bash
xcode-select --install
```

Install Homebrew:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Install Kubernetes/Helm tooling:

```bash
brew install kubectl k3d helm
```

Install Docker Desktop:

```bash
brew install --cask docker
```

Then launch Docker Desktop and wait until it is fully started.

Install Conda:

```bash
brew install --cask anaconda
```

## 2. Verify Tooling

```bash
docker --version
kubectl version --client
k3d version
helm version
conda --version
make --version
```

## 3. Clone Repository

```bash
git clone https://github.com/kmikol/ml-system.git
cd ml-system
```

## 4. Create and Activate Conda Environment

```bash
conda create -n ml-system python=3.11 -y
conda activate ml-system
python -m pip install --upgrade pip
```

## 5. Install Python Dependencies

Install dependencies from the single root entrypoint:

```bash
pip install -r requirements.txt
```

This includes python dependancies for functionality running outside of the kubernetes. E.g. tests, building documentation etc.


