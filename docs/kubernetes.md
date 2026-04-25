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

## One-Time Minikube Start

From the repository root on EC2:

```bash
minikube start --driver=docker --ports=30080:30080
kubectl get nodes
```

The `--ports` flag publishes the Kubernetes NodePort from the Minikube Docker container onto the EC2 host. If Minikube is already running without this port mapping, recreate it before the demo:

```bash
minikube delete
minikube start --driver=docker --ports=30080:30080
```

## Build and Load the API Image

The API image packages the inference code plus the active model files from `models/`.

```bash
docker build -f Dockerfile.app -t mediwatch-api:local .
minikube image load mediwatch-api:local
```

## Optional S3 Secret

If the API should download promoted models from S3, create the secret after copying the example file:

```bash
cp k8s/secret.example.yaml k8s/secret.yaml
# edit k8s/secret.yaml with the real bucket name
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secret.yaml
```

Do not commit `k8s/secret.yaml`.

## Deploy the API

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/api-service.yaml
```

Check rollout status:

```bash
kubectl -n mediwatch rollout status deployment/mediwatch-api
kubectl -n mediwatch get pods
kubectl -n mediwatch get svc mediwatch-api
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
kubectl -n mediwatch rollout restart deployment/mediwatch-api
kubectl -n mediwatch rollout status deployment/mediwatch-api
```

This manual sequence is the baseline that the CI/CD section will automate.
