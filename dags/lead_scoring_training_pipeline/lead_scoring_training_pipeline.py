from airflow import DAG
"""
Lead Scoring Training Pipeline Module
This module defines an Apache Airflow DAG that orchestrates the training pipeline
for a Lead Scoring System. The pipeline is organized into three sequential tasks:
1. Start MLflow Server:
    Runs a Python callable that checks whether the MLflow server is running, and starts
    it if it is not. This step ensures that MLflow tracking is available for subsequent
    tasks.
2. Encode Categorical Variables:
    Processes and encodes categorical features required for model training. The Python
    callable associated with this task handles the feature transformation necessary for
    the machine learning workflow.
3. Train Model:
    Executes the model training process using the processed data and defined configurations.
    The task runs a Python callable that returns the trained model for further evaluation or
    deployment.
The DAG is configured with the following default settings:
- Owner: airflow
- Start Date: January 1, 2025
- Retries: 0 (without automatic retry on failure)
- Retry Delay: 5 seconds
The pipeline is scheduled to run daily and does not perform catchup on missed intervals.
Tags are assigned to help with categorizing the DAG under model training operations.
Usage:
     The DAG is instantiated within a context manager and tasks are linked using bitshift
     operators to define the execution order: start MLflow server -> encode features -> train model.
"""
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from lead_scoring_training_pipeline.utils import (
    start_mlflow_server_if_not_running,
    encode_features,
    get_trained_model,
)

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 1, 1),
    'retries': 0,
    'retry_delay': timedelta(seconds=5),
}

with DAG(
    dag_id='Lead_scoring_training_pipeline',
    default_args=default_args,
    description='Training pipeline for Lead Scoring System',
    schedule_interval='@daily',
    catchup=False,
    tags=['model training']
) as dag:
    # Task: Start MLflow server (if not already running)
    start_mlflow = PythonOperator(
        task_id='start_mlflow_server',
        python_callable=start_mlflow_server_if_not_running,
        doc_md="""\
        **Task:** Start MLflow Server
        **Description:** Checks if the MLflow server is running and starts it if necessary.
        """
    )

    # Task: Encode categorical variables
    encoding_categorical_variables = PythonOperator(
        task_id='encoding_categorical_variables',
        python_callable=encode_features,
        doc_md="""\
        **Task:** Encode Categorical Variables
        **Description:** Processes and encodes the categorical features required for model training.
        """
    )

    # Task: Train model
    training_model = PythonOperator(
        task_id='training_model',
        python_callable=get_trained_model,
        doc_md="""\
        **Task:** Train Model
        **Description:** Executes the model training process using the processed data.
        """
    )

    # Define the task pipeline
    start_mlflow >> encoding_categorical_variables >> training_model
