# NLP Data Processing & Matching App

## Overview
This project is a Flask-based web application that demonstrates **NLP-driven data matching** between two datasets.  

It uses:
- **TF-IDF vectorization** + cosine similarity for pre-filtering potential matches.  
- **OpenAI GPT models** to refine matches and generate structured JSON reasoning.  
- **ETL-style preprocessing** pipelines for preparing and cleaning data.  
- **Prometheus + Grafana monitoring** for production health and metrics.  
- **GitHub Actions** for CI/CD automated deployment to an EC2 server.  
- **Docker + docker-compose** to package and deploy the app consistently.  

The app exposes a simple **web form** where you enter how many rows to process. It downloads source + lookup data, applies matching, and produces a downloadable CSV with entity match results.

---

## Features
- Uploads or pulls sample datasets and compares records.  
- Combines multiple matching columns with configurable weights.  
- Runs TF-IDF similarity to rank top candidate matches.  
- Sends structured prompts to GPT for predictive reasoning + ranking.  
- Produces downloadable CSV of matches with reasoning.  
- Provides a **`/status`** endpoint for health checks.  
- Automatically exposes Prometheus metrics for monitoring.  

---

## How to Run

### 1. Clone the repo
```bash
git clone https://github.com/your-username/Portfolio.git
cd Portfolio/nlp_app
```

### 2. Build and run with Docker Compose
```bash
docker compose up -d --build
```

This launches three services:  
- `nlp_app` → Flask web app (port 8000)  
- `prometheus` → Monitoring backend (port 9090)  
- `grafana` → Dashboards (port 3000)  

### 3. Access the services
- Web app: `http://<EC2_PUBLIC_IP>:8000`  
- Prometheus: `http://<EC2_PUBLIC_IP>:9090`  
- Grafana: `http://<EC2_PUBLIC_IP>:3000` (login: `admin` / `admin` by default)

---

## Repository Structure
```
.
├── .github/workflows/deploy.yml     # GitHub Actions CI/CD pipeline
├── docker-compose.yml               # Defines app + Prometheus + Grafana stack
├── prometheus.yml                   # Prometheus scrape configuration
├── nlp_app/                         # Main Flask application
│   ├── app.py                       # Core app logic (Flask routes + NLP functions)
│   ├── new_nlp_pull.py              # Data pull helper
│   ├── requirements.txt             # Python dependencies
│   ├── templates/index.html         # Web form frontend
│   └── Dockerfile                   # Container definition
├── legacy/                          # Old experiments (BigQuery, Redshift, SageMaker)
└── README.md                        # (this file)
```

---

## How to Use the App
1. Navigate to your web app (`:8000`).  
2. Enter a number of rows to process (1–1000).  
3. Click **Process Data & Download CSV**.  
4. Wait for the app to run TF-IDF + GPT matching.  
5. A CSV file will download with:  
   - lookup record  
   - candidate matches  
   - GPT decision (same entity, ranking, reasoning)  

---

## Monitoring & Health
- **Prometheus** automatically scrapes `/metrics` from the Flask app.  
- **Grafana** dashboards can be added for request latency, error rate, and throughput.  
- Health endpoints:  
  - `/status` → JSON response confirming app is running  
  - `/metrics` → Prometheus metrics  

---

## CI/CD Deployment
- Pushing to `main` triggers GitHub Actions.  
- The workflow (`.github/workflows/deploy.yml`) connects via SSH to the EC2 instance.  
- It pulls the latest code and restarts Docker Compose automatically.  

---

## Tech Stack
- **Python / Flask** (app framework)  
- **scikit-learn, pandas, numpy** (data processing + NLP)  
- **OpenAI Python SDK** (LLM-based reasoning)  
- **Docker / docker-compose** (containerized deployment)  
- **GitHub Actions** (CI/CD automation)  
- **Prometheus + Grafana** (monitoring + visualization)  
