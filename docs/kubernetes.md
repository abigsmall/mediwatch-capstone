# Kubernetes Deployment

This capstone uses Kubernetes for the FastAPI inference service and Streamlit UI. The rest of the local demo stack runs with Docker Compose.

The current Compose stack still defines `api` and `ui` services for local development. For the Kubernetes requirement, use the Kubernetes endpoints on ports `30080` and `30051` as the demo endpoints.

## Target Setup

- EC2 runs Docker, Docker Compose, and Minikube.
- Docker Compose continues to run Airflow, MLflow, and Postgres.
- Minikube runs the `mediwatch-api` and `mediwatch-ui` Kubernetes Deployments and NodePort Services.
- The EC2 security group opens TCP ports `30080` and `30051` for evaluator access.

```text
EC2
├── Docker Compose
│   ├── airflow-webserver
│   ├── airflow-scheduler
│   ├── mlflow
│   └── postgres
└── Minikube Kubernetes
    ├── mediwatch-api
    └── mediwatch-ui
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
minikube start --driver=docker --ports=30080:30080 --ports=30051:30051
minikube kubectl -- get nodes
```

## Build and Load the App Image

The app image packages the API, UI, and active model files from `models/`. Both Kubernetes Deployments use this same image with different commands.

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

## Deploy the API and UI

```bash
minikube kubectl -- apply -f k8s/namespace.yaml
minikube kubectl -- apply -f k8s/api-deployment.yaml
minikube kubectl -- apply -f k8s/api-service.yaml
minikube kubectl -- apply -f k8s/ui-deployment.yaml
minikube kubectl -- apply -f k8s/ui-service.yaml
```

Check rollout status:

```bash
minikube kubectl -- -n mediwatch rollout status deployment/mediwatch-api
minikube kubectl -- -n mediwatch rollout status deployment/mediwatch-ui
minikube kubectl -- -n mediwatch get pods
minikube kubectl -- -n mediwatch get svc
```

## Test From EC2

```bash
curl http://127.0.0.1:30080/health
curl http://$(minikube ip):30080/health
curl http://127.0.0.1:30051/_stcore/health
```

## Test From Outside EC2

Open TCP ports `30080` and `30051` in the EC2 instance security group, then call:

```bash
curl http://<ec2-public-ip>:30080/health
curl http://<ec2-public-ip>:30051/_stcore/health
```

Swagger docs and the Streamlit UI are available at:

```text
http://<ec2-public-ip>:30080/docs
http://<ec2-public-ip>:30051
```

## Redeploy After Retraining

After retraining promotes a new model:

```bash
docker build -f Dockerfile.app -t mediwatch-api:local .
minikube image load mediwatch-api:local
minikube kubectl -- -n mediwatch rollout restart deployment/mediwatch-api
minikube kubectl -- -n mediwatch rollout restart deployment/mediwatch-ui
minikube kubectl -- -n mediwatch rollout status deployment/mediwatch-api
minikube kubectl -- -n mediwatch rollout status deployment/mediwatch-ui
```

This manual sequence is the baseline that the CI/CD section will automate.
