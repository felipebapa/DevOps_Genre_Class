# Monitoring

This project uses Prometheus and Grafana through the
`kube-prometheus-stack` Helm chart.

## Components

The monitoring stack includes:

- Prometheus
- Grafana
- kube-state-metrics
- node-exporter
- Prometheus Operator

Alertmanager is disabled because alerting is outside the scope of this project.

## Installation

Create the monitoring namespace:

```bash
kubectl create namespace monitoring