# Monitoring

This project uses Prometheus and Grafana through the `kube-prometheus-stack` Helm chart to monitor the Kubernetes environment where the Genre Classifier application is deployed.

The monitoring solution provides visibility into:

- CPU usage
- Memory usage
- Pod readiness
- Kubernetes workload status
- Node-level metrics

## Components

The monitoring stack includes:

- Prometheus
- Grafana
- kube-state-metrics
- node-exporter
- Prometheus Operator

Alertmanager is disabled because alerting is outside the current scope of the project.

## Prerequisites

Before deploying the monitoring stack, the following components must already be available:

- A working Kubernetes cluster
- `kubectl` configured to access the cluster
- Helm 3 installed
- Network connectivity to download Helm charts and container images

## Installation

### 1. Create the monitoring namespace

```bash
kubectl create namespace monitoring
```

Verify that the namespace was created:

```bash
kubectl get namespaces
```

### 2. Add the Prometheus Community Helm repository

```bash
helm repo add prometheus-community \
  https://prometheus-community.github.io/helm-charts
```

Update the local Helm repository information:

```bash
helm repo update
```

### 3. Install kube-prometheus-stack

The project includes a lightweight Helm configuration in:

```text
monitoring/values.yaml
```

Install the stack with:

```bash
helm install monitoring \
  prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --values monitoring/values.yaml
```

### 4. Verify the deployment

Check the monitoring pods:

```bash
kubectl get pods -n monitoring
```

The main components should eventually reach a `Running` state.

Typical resources include:

```text
monitoring-grafana-...
monitoring-kube-prometheus-operator-...
monitoring-kube-state-metrics-...
monitoring-prometheus-node-exporter-...
prometheus-monitoring-kube-prometheus-prometheus-0
```

## Resource Optimization

The monitoring configuration is intentionally lightweight because the Kubernetes cluster is running in a lab environment with limited resources.

The configuration includes:

- Prometheus retention limited to 2 days
- Reduced CPU and memory requests and limits
- Grafana persistence disabled
- Alertmanager disabled
- kube-state-metrics enabled
- node-exporter enabled

These settings are defined in:

```text
monitoring/values.yaml
```

## Grafana Access

Grafana is not exposed publicly by default.

For local access, run:

```bash
kubectl port-forward \
  -n monitoring \
  svc/monitoring-grafana \
  3000:80
```

Then open:

```text
http://localhost:3000
```

If Grafana is running on a remote Kubernetes host accessed through SSH, an additional SSH tunnel can be used from the local workstation:

```bash
ssh -L 3000:localhost:3000 <user>@<kubernetes-host>
```

The default Grafana administrator username is:

```text
admin
```

The administrator password is stored in the Kubernetes Secret `monitoring-grafana`.

Retrieve it with:

```bash
kubectl get secret monitoring-grafana \
  -n monitoring \
  -o jsonpath='{.data.admin-password}' | base64 -d; echo
```

The Grafana password is intentionally not stored in this repository.

## Custom Genre Classifier Dashboard

A custom Grafana dashboard was created to monitor the Genre Classifier application.

The dashboard includes three main panels:

- CPU Usage
- Memory Usage
- Pod Status

The queries use stable pod-name matching instead of a specific pod name so that the dashboard continues to work after Kubernetes rolling deployments.

The application pods are matched with:

```promql
pod=~"genre-classifier-.*"
```

This prevents the dashboard from breaking when Kubernetes creates a replacement pod with a new generated name.

## CPU Usage

The CPU panel uses the following PromQL query:

```promql
sum(
  rate(
    container_cpu_usage_seconds_total{
      namespace="default",
      pod=~"genre-classifier-.*",
      container="genre-classifier"
    }[5m]
  )
) * 1000
```

The query:

- Selects the Genre Classifier container
- Calculates the CPU usage rate over a 5-minute window
- Sums all matching time series
- Converts CPU cores into millicores

The dashboard displays the result using:

```text
mCPU
```

For example:

```text
3 mCPU
```

is approximately equivalent to:

```text
0.003 CPU cores
```

## Memory Usage

The memory panel uses the following PromQL query:

```promql
sum(
  container_memory_working_set_bytes{
    namespace="default",
    pod=~"genre-classifier-.*",
    container="genre-classifier"
  }
)
```

This metric represents the active working set memory used by the application container.

Grafana displays this value using IEC memory units such as:

```text
MiB
GiB
```

## Pod Status

The Pod Status panel uses the following query:

```promql
max(
  kube_pod_status_ready{
    namespace="default",
    pod=~"genre-classifier-.*",
    condition="true"
  }
)
```

The result is interpreted as:

```text
1 = Ready
0 = Not Ready
```

Grafana value mapping is configured to display:

```text
1 -> Ready
0 -> Not Ready
```

The panel uses a green visual state when the application is ready.

Using `max()` is useful during rolling deployments because Kubernetes can temporarily have both the old and new application pods running at the same time.

If at least one matching pod is ready, the dashboard indicates that the application is still available.

## Monitoring Validation

The monitoring environment was validated by confirming that:

- Prometheus successfully collects Kubernetes metrics
- Grafana uses Prometheus as its data source
- kube-state-metrics provides Kubernetes object state information
- node-exporter provides node-level metrics
- The Genre Classifier CPU usage is visible
- The Genre Classifier memory usage is visible
- The Genre Classifier pod readiness is visible
- Metrics continue working after Kubernetes rolling deployments

## Security Considerations

No monitoring credentials or secrets are stored in this repository.

Sensitive values such as the Grafana administrator password are stored inside Kubernetes Secrets and retrieved only when required.

The following information must never be committed to the repository:

- Grafana passwords
- Kubernetes bearer tokens
- kubeconfig files containing credentials
- Docker registry credentials
- Jenkins credentials

The repository only stores non-sensitive deployment and monitoring configuration.

## Useful Commands

Check monitoring pods:

```bash
kubectl get pods -n monitoring
```

Check monitoring services:

```bash
kubectl get services -n monitoring
```

Check the Prometheus pod:

```bash
kubectl get pods -n monitoring | grep prometheus
```

Check the Grafana pod:

```bash
kubectl get pods -n monitoring | grep grafana
```

Restart the Grafana deployment if required:

```bash
kubectl rollout restart deployment/monitoring-grafana \
  -n monitoring
```

Verify Grafana rollout status:

```bash
kubectl rollout status deployment/monitoring-grafana \
  -n monitoring
```

## Uninstallation

The monitoring stack can be removed with:

```bash
helm uninstall monitoring \
  --namespace monitoring
```

The namespace can then be removed with:

```bash
kubectl delete namespace monitoring
```

## Project Structure

The monitoring files are organized as follows:

```text
monitoring/
├── README.md
└── values.yaml
```

`values.yaml` contains the Helm configuration used for the lightweight monitoring deployment.

`README.md` documents the installation, access, validation, queries, and security considerations of the monitoring solution.
