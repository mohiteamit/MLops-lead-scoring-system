##############################################################################
# Import necessary modules
##############################################################################
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
from lead_scoring_training_pipeline.utils import start_mlflow_server_if_not_running, encode_features, get_trained_model

##############################################################################
# Define default arguments and DAG
##############################################################################
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 1, 1),
    'retries': 0,
    'retry_delay': timedelta(seconds=5)
}

ML_training_dag = DAG(
    dag_id='Lead_scoring_training_pipeline',
    default_args=default_args,
    description='Training pipeline for Lead Scoring System',
    schedule_interval='@daily',
    catchup=False,
    tags=['model training']
)

##############################################################################
# Task: Start MLflow Server (if not already running)
##############################################################################
start_mlflow = PythonOperator(
    task_id='start_mlflow_server',
    python_callable=start_mlflow_server_if_not_running,
    dag=ML_training_dag
)

##############################################################################
# Task: Encode categorical variables
##############################################################################
encoding_categorical_variables = PythonOperator(
    task_id='encoding_categorical_variables',
    python_callable=encode_features,
    dag=ML_training_dag
)

##############################################################################
# Task: Train model
##############################################################################
training_model = PythonOperator(
    task_id='training_model',
    python_callable=get_trained_model,
    dag=ML_training_dag
)

##############################################################################
# Define task dependencies
##############################################################################
start_mlflow >> encoding_categorical_variables >> training_model
