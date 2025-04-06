from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

# Set the path so Airflow can import your pipeline code
sys.path.insert(0, os.path.abspath("/home/airflow/dags"))

# Import test functions
import unit_test.test_with_pytest as test_functions

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 0,
}

with DAG(
    dag_id='unit_test_runner',
    default_args=default_args,
    description='Run unit tests for data pipeline functions',
    schedule_interval=None,  # Trigger manually or change to '@daily'
    catchup=False,
    tags=['testing', 'validation']
) as dag:

    test_load = PythonOperator(
        task_id='test_load_data_into_db',
        python_callable=test_functions.test_load_data_into_db
    )

    test_city_tier = PythonOperator(
        task_id='test_map_city_tier',
        python_callable=test_functions.test_map_city_tier
    )

    test_cat_vars = PythonOperator(
        task_id='test_map_categorical_vars',
        python_callable=test_functions.test_map_categorical_vars
    )

    test_interactions = PythonOperator(
        task_id='test_interactions_mapping',
        python_callable=test_functions.test_interactions_mapping
    )

    # Set task execution order
    test_load >> test_city_tier >> test_cat_vars >> test_interactions
