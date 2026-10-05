# Iris Data Pipeline

## What it does

Learn Iris classification and the steps around a small data pipeline.
The local dashboard reads the bundled measurements for 150 flowers, lets
you explore the three species, and fits a random-forest classifier to predict
species from sepal and petal measurements. It displays evaluation results,
including a confusion matrix, ROC curves, feature importance and cross-validation.

The repository also contains a larger data-engineering example:

```text
Iris CSV → PostgreSQL → dbt transformations and checks
         → model training and MLflow experiment records → reports and dashboard
```

Airflow schedules that workflow. PostgreSQL holds the tables, dbt organizes
and checks them, and MLflow records model experiments. These components
illustrate how analysis can be repeated and tracked; they are optional for
trying the local machine-learning dashboard.

Start with the standalone Streamlit example below. It works from the bundled
CSV without running the Docker services. The local app was checked; the full
PostgreSQL/dbt/Airflow/MLflow stack has not been verified end to end.

## Input

Input: 150 bundled rows with `sepal_length,sepal_width,petal_length,petal_width,species`; the four measurement columns are numeric.

## Output

Output: interactive tables and plots in the browser, including random-forest classification metrics, confusion matrix, ROC curves, feature importance and five-fold cross-validation. This dashboard route does not save a model or prediction CSV. Check that the data explorer shows 150 observations and three species, with 50 rows each, and the ML Analysis tab shows classification results.

## Try it

### Start here: local machine-learning demo

To explore Iris classification, run the existing dashboard directly on the bundled CSV. Use Python 3.11 in a separate environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app/app.py
```

Run these commands from the repository root after cloning it. Open the local URL printed by Streamlit, usually `http://localhost:8501`. The existing dashboard falls back to `data/iris.csv` when no database is available. This local route needs no Docker, PostgreSQL, dbt, Airflow or MLflow service.

The unchanged dashboard was checked using Streamlit AppTest on the bundled CSV: no app exceptions, 13 Plotly charts, about 9 seconds including app initialization on the test machine. Installation and browser startup take additional time. `requirements.txt` records the checked package versions; the existing old Plotly version failed with NumPy 2, so this local environment uses NumPy 1.26.4. This is a local example check, not a fresh-environment installation test or a guarantee about other input files.

### Optional: data-engineering pipeline

The remaining sections describe the larger Docker example: PostgreSQL stores the data, dbt transforms and checks it, Airflow schedules the steps, and MLflow records model experiments. Use it when learning these engineering components. It is optional for the local classification demo; complete Docker-stack execution remains unverified.

### Architecture

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

### Components

| Component | Role | Files |
|---|---|---|
| **PostgreSQL** | Data storage: create table, import CSV | `init.sql` |
| **dbt** | Data transformation + layered modeling + testing | `dbt_project/models/` |
| **Airflow** | Orchestration: schedule and trigger pipeline | `airflow/dags/iris_pipeline.py` |
| **MLflow** | ML experiment tracking: log params, metrics, charts | `ml/train_model.py` |
| **Python** | Static reports (CSV + PNG) | `python_visual/plot_iris.py` |
| **Streamlit** | Interactive dashboard (Data Explorer + ML Analysis) | `streamlit_app/app.py` |

### dbt Model Layers

```
iris (raw table)
  → staging/stg_iris              clean + standardize
    → marts/dim_species            dimension: species info
    → marts/fct_measurements       fact: each record + computed fields
      → marts/mart_species_summary metrics: stable BI interface
```

### Project Structure

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

### Docker Quick Start (optional)

```bash
# 1. Clone and set up environment
git clone https://github.com/yujuan-zhang/End2End-iris.git
cd End2End-iris
cp .env.example .env  # edit passwords if needed

# 2. Start all services
docker compose up -d --build

# 3. Open web interfaces
# Streamlit Dashboard:  http://localhost:8501
# Airflow Scheduler:    http://localhost:8080  (admin / changeme from .env.example)
# MLflow Tracking:      http://localhost:5050

# 4. Trigger the pipeline
# Go to Airflow → iris_pipeline → Trigger DAG

# 5. Check logs
docker compose logs dbt
docker compose logs airflow
docker compose logs mlflow

# 6. Shut down
docker compose down
```

### Input, output and first-run checks

`data/iris.csv` contains 150 observations with the header `sepal_length,sepal_width,petal_length,petal_width,species`. The four measurements are numeric; species is the class label. PostgreSQL imports this CSV only when its data volume is initialized. Editing the CSV does not automatically reload an existing volume.

The dbt models create `stg_iris`, `dim_species`, `fct_measurements` and `mart_species_summary`. The report writes `results/iris_summary.csv`, `results/bar_chart.png` and `results/scatter_plot.png`. Airflow schedules the pipeline; MLflow stores experiments in its Docker volume. These are different outputs from `data/result.csv`.

After startup, check `docker compose logs dbt` for a successful run and all data-quality tests passing; then check that the three report files exist and are non-empty. Inspect service logs if a UI is not ready immediately. The Airflow password is `AIRFLOW_ADMIN_PASSWORD` from `.env`; change it there before first startup if desired. Normal shutdown preserves volumes. Destructive reset is a separate operation and is not needed for daily use.

The Compose configuration and CSV contract were checked. Full image builds, Airflow execution and end-to-end runtime have not been measured in this repair pass; no runtime estimate is claimed. Existing host services can conflict with ports 5432, 5050, 8080 and 8501.

### Streamlit Dashboard

The dashboard has two tabs:

- **Data Explorer** — interactive scatter plots, bar charts, statistics cards
- **ML Analysis** — confusion matrix, ROC curves, feature importance, PCA, violin plots, cross-validation

### MLflow Experiment Tracking

- Compare different model runs with varying hyperparameters
- View 6 charts per run: confusion matrix, ROC curves, feature importance, PCA scatter, violin plots, CV radar
- Access at http://localhost:5050

