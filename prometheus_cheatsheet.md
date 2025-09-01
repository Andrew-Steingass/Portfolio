# Prometheus Cheat Sheet for NLP Flask App

Your Flask app automatically exposes metrics at `/metrics` thanks to `prometheus_flask_exporter`.  
Prometheus scrapes these and stores them for querying.

---

## 1. Access Prometheus
- URL: `http://<server-ip>:9090`
- Key tabs:
  - **Status → Targets** → verify `flask_app` target is UP.
  - **Graph** → run queries to see app metrics.

---

## 2. Key Metrics for Your App

### Request Metrics
- `flask_http_request_total`  
  Counter of all HTTP requests by endpoint, method, and status code.  
  Example:  
  ```promql
  sum by (status) (rate(flask_http_request_total[1m]))
  ```
  → Requests per second grouped by status (200, 500, etc.).

- `flask_http_request_duration_seconds_count`  
  Number of requests tracked for latency.

- `flask_http_request_duration_seconds_sum`  
  Total request duration in seconds.  
  To calculate **average latency**:  
  ```promql
  rate(flask_http_request_duration_seconds_sum[5m]) 
    / rate(flask_http_request_duration_seconds_count[5m])
  ```

### Error Metrics
- `flask_http_request_exceptions_total`  
  Total number of exceptions raised.  
  ```promql
  rate(flask_http_request_exceptions_total[5m])
  ```

### Response Metrics
- `flask_http_request_latency_seconds` (if enabled)  
  Histogram of request durations by bucket.  
  ```promql
  histogram_quantile(0.95, rate(flask_http_request_latency_seconds_bucket[5m]))
  ```
  → 95th percentile request latency.

---

## 3. Process Metrics
These come from Python runtime:
- `process_cpu_seconds_total` → total CPU time used by the app.
- `process_resident_memory_bytes` → memory usage.
- `python_gc_objects_collected_total` → garbage-collected objects.

---

## 4. Example Queries for Your NLP App
- Requests per second to `/process`:  
  ```promql
  rate(flask_http_request_total{endpoint="/process"}[1m])
  ```

- Error rate (5xx responses):  
  ```promql
  rate(flask_http_request_total{status=~"5.."}[5m])
  ```

- Average latency for `/process`:  
  ```promql
  rate(flask_http_request_duration_seconds_sum{endpoint="/process"}[5m]) 
    / rate(flask_http_request_duration_seconds_count{endpoint="/process"}[5m])
  ```

- Memory usage:  
  ```promql
  process_resident_memory_bytes
  ```

---

## 5. Next Steps
- Use Grafana (`http://<server-ip>:3000`) to build dashboards from these metrics.
- Add alerts in `prometheus.yml` (e.g., alert if error rate > 5%).
