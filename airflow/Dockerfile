FROM apache/airflow:2.8.1-python3.11

# 安装 dbt 和 Python 依赖（在 airflow 用户下）
COPY requirements.txt .
RUN pip install -r requirements.txt
