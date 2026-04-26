# Kubernetes Deployment

This capstone uses Kubernetes for the FastAPI inference service only. The rest of the local demo stack runs with Docker Compose.

The current Compose stack still defines an `api` service on port `8000`. For the Kubernetes requirement, use the Kubernetes endpoint on port `30080` as the demo endpoint.

## Target Setup

- EC2 runs Docker, Docker Compose, and Minikube.
- Docker Compose continues to run Airflow, MLflow, Postgres, and optionally the Streamlit UI.
- Minikube runs the `mediwatch-api` Kubernetes Deployment and NodePort Service.
- The EC2 security group opens TCP port `30080` for evaluator access.

```text
EC2
├── Docker Compose
│   ├── airflow-webserver
│   ├── airflow-scheduler
│   ├── mlflow
│   ├── postgres
│   └── ui
└── Minikube Kubernetes
    └── mediwatch-api
```

## Fresh EC2 Prerequisites

On a fresh EC2 instance, `minikube` may not be available through the default Ubuntu apt repositories:

```bash
sudo apt install minikube
# E: Unable to locate package minikube
```

If that happens, install Minikube from the official Linux package URL instead:

```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube_latest_amd64.deb
sudo dpkg -i minikube_latest_amd64.deb
minikube version
```

If `minikube start --driver=docker` fails with a Docker permission error, the current user probably cannot access the Docker daemon socket yet. Add the user to the `docker` group, then refresh the shell session:

```bash
sudo usermod -aG docker $USER
newgrp docker
docker ps
```

If `newgrp docker` does not refresh the permissions cleanly, log out of the EC2 SSH session and log back in. You may not have to do this, but I actually had to restart the EC2 instance before the change had any effect.

Avoid running Minikube itself with `sudo`; the Docker driver should work as the normal user once Docker group membership is active.

Standalone `kubectl` was not installed for me, so you may need to use Minikube's bundled `kubectl`:

```bash
minikube kubectl -- get pods -A
```

## One-Time Minikube Start

From the repository root on EC2:

```bash
minikube start --driver=docker --ports=30080:30080
minikube kubectl -- get nodes
```

## Build and Load the API Image

The API image packages the inference code plus the active model files from `models/`.

```bash
docker build -f Dockerfile.app -t mediwatch-api:local .
minikube image load mediwatch-api:local
```

## Optional S3 Secret

<!-- TODO: Decide whether to keep this section as-is or modify it -->

If the API should download promoted models from S3, create the secret after copying the example file:

```bash
cp k8s/secret.example.yaml k8s/secret.yaml
# edit k8s/secret.yaml with the real bucket name
minikube kubectl -- apply -f k8s/namespace.yaml
minikube kubectl -- apply -f k8s/secret.yaml
```

Do not commit `k8s/secret.yaml`.

## Deploy the API

```bash
minikube kubectl -- apply -f k8s/namespace.yaml
minikube kubectl -- apply -f k8s/api-deployment.yaml
minikube kubectl -- apply -f k8s/api-service.yaml
```

Check rollout status:

```bash
minikube kubectl -- -n mediwatch rollout status deployment/mediwatch-api
minikube kubectl -- -n mediwatch get pods
minikube kubectl -- -n mediwatch get svc mediwatch-api
```

## Test From EC2

```bash
curl http://127.0.0.1:30080/health
curl http://$(minikube ip):30080/health
```

## Test From Outside EC2

Open TCP port `30080` in the EC2 instance security group, then call:

```bash
curl http://<ec2-public-ip>:30080/health
```

Swagger docs are available at:

```text
http://<ec2-public-ip>:30080/docs
```

## Redeploy After Retraining

After retraining promotes a new model:

```bash
docker build -f Dockerfile.app -t mediwatch-api:local .
minikube image load mediwatch-api:local
minikube kubectl -- -n mediwatch rollout restart deployment/mediwatch-api
minikube kubectl -- -n mediwatch rollout status deployment/mediwatch-api
```

This manual sequence is the baseline that the CI/CD section will automate.
