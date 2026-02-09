# Iris Data Pipeline

End-to-end data pipeline with Docker: PostgreSQL + dbt + Airflow + MLflow + Streamlit, using the Iris dataset.

## Architecture

```
iris.csv
  ↓  PostgreSQL: create table + import
iris table
  ↓  dbt: Staging — clean + classify
stg_iris
  ↓  dbt: Marts — dimension + fact + metrics
dim_species / fct_measurements / mart_species_summary
  ↓  dbt test: 13 data quality checks
  ↓  Python: static report             → results/ (CSV + PNG)
  ↓  MLflow: train model + log metrics  → http://localhost:5050
  ↓  Streamlit: interactive dashboard   → http://localhost:8501

Airflow orchestrates the above → http://localhost:8080
```

## Components

| Component | Role | Files |
|---|---|---|
| **PostgreSQL** | Data storage: create table, import CSV | `init.sql` |
| **dbt** | Data transformation + layered modeling + testing | `dbt_project/models/` |
| **Airflow** | Orchestration: schedule and trigger pipeline | `airflow/dags/iris_pipeline.py` |
| **MLflow** | ML experiment tracking: log params, metrics, charts | `ml/train_model.py` |
| **Python** | Static reports (CSV + PNG) | `python_visual/plot_iris.py` |
| **Streamlit** | Interactive dashboard (Data Explorer + ML Analysis) | `streamlit_app/app.py` |

## dbt Model Layers

```
iris (raw table)
  → staging/stg_iris              clean + standardize
    → marts/dim_species            dimension: species info
    → marts/fct_measurements       fact: each record + computed fields
      → marts/mart_species_summary metrics: stable BI interface
```

## Project Structure

```
├── .env.example                    # Environment variables template
├── docker-compose.yml              # All services definition
├── init.sql                        # PostgreSQL init script
├── data/
│   └── iris.csv                    # Raw data
├── dbt_project/
│   ├── dbt_project.yml
│   ├── profiles.yml
│   └── models/
│       ├── staging/
│       │   └── stg_iris.sql        # Staging: data cleaning
│       ├── marts/
│       │   ├── dim_species.sql     # Dimension table
│       │   ├── fct_measurements.sql # Fact table
│       │   └── mart_species_summary.sql # Metrics layer
│       └── schema.yml              # Data tests (13 tests)
├── airflow/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── dags/
│       └── iris_pipeline.py        # DAG: dbt → test → report + ML
├── ml/
│   └── train_model.py              # Train model + log to MLflow
├── python_visual/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── plot_iris.py                # Generate static reports
├── streamlit_app/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py                      # Interactive dashboard
└── results/                        # Output: CSV + PNG (gitignored)
```

## Quick Start

```bash
# 1. Clone and set up environment
git clone <repo-url>
cd iris-data-pipeline
cp .env.example .env  # edit passwords if needed

# 2. Start all services
docker compose up -d --build

# 3. Open web interfaces
# Streamlit Dashboard:  http://localhost:8501
# Airflow Scheduler:    http://localhost:8080  (admin / admin)
# MLflow Tracking:      http://localhost:5050

# 4. Trigger the pipeline
# Go to Airflow → iris_pipeline → Trigger DAG

# 5. Check logs
docker compose logs dbt
docker compose logs airflow
docker compose logs mlflow

# 6. Shut down
docker compose down -v
```

## Streamlit Dashboard

The dashboard has two tabs:

- **Data Explorer** — interactive scatter plots, bar charts, statistics cards
- **ML Analysis** — confusion matrix, ROC curves, feature importance, PCA, violin plots, cross-validation

## MLflow Experiment Tracking

- Compare different model runs with varying hyperparameters
- View 6 charts per run: confusion matrix, ROC curves, feature importance, PCA scatter, violin plots, CV radar
- Access at http://localhost:5050
