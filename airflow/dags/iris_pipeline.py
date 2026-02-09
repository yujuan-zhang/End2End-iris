from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# DAG 默认参数
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# 定义 DAG（调度流程）
with DAG(
    'iris_pipeline',
    default_args=default_args,
    description='Iris data pipeline: dbt -> test -> report + ML training',
    schedule='@daily',           # 每天执行一次
    start_date=datetime(2024, 1, 1),
    catchup=False,               # 不补跑历史任务
) as dag:

    # 任务1: dbt run — 执行数据转换
    dbt_run = BashOperator(
        task_id='dbt_run',
        bash_command='cd /opt/dbt && dbt run --profiles-dir /opt/dbt',
    )

    # 任务2: dbt test — 数据质量检查
    dbt_test = BashOperator(
        task_id='dbt_test',
        bash_command='cd /opt/dbt && dbt test --profiles-dir /opt/dbt',
    )

    # 任务3: 调用独立脚本生成报告
    python_report = BashOperator(
        task_id='python_report',
        bash_command='python /opt/python_visual/plot_iris.py',
    )

    # 任务4: 训练 ML 模型，记录到 MLflow
    ml_train = BashOperator(
        task_id='ml_train',
        bash_command='python /opt/ml/train_model.py',
    )

    # 定义执行顺序:
    # dbt_run → dbt_test → python_report
    #                     → ml_train (与 python_report 并行)
    dbt_run >> dbt_test >> [python_report, ml_train]
