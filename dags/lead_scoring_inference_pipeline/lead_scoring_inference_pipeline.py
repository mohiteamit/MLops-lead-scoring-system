"""
Lead Scoring Inference Pipeline DAG

This module defines the Airflow DAG for the Lead Scoring Inference Pipeline.
It performs the following steps:
1. Starts the MLflow server if not running.
2. Validates the raw data schema.
3. Loads data into the database.
4. Maps city tier, categorical variables, and feature interactions.
5. Cleans up interim database data.
6. Validates model input schema.
7. Encodes categorical variables.
8. Generates model predictions.
9. Checks the prediction ratio.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# Import functions for the inference pipeline tasks.
from lead_scoring_data_pipeline.utils import (
    load_data_into_db, map_city_tier, map_categorical_vars, interactions_mapping, clean_up_db
)
from lead_scoring_data_pipeline.data_validation_checks import (
    raw_data_schema_check, model_input_schema_check
)
from lead_scoring_training_pipeline.utils import (
    start_mlflow_server_if_not_running, encode_features
)
from lead_scoring_inference_pipeline.utils import (
    get_models_prediction, prediction_ratio_check
)
from lead_scoring_inference_pipeline.schema import (
    RAW_DATA_SCHEMA, MODEL_INPUT_SCHEMA
)
from lead_scoring_inference_pipeline.constants import CSV_DATA, TABLE_NAME

# Default arguments for the DAG tasks.
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 1, 1),
    'retries': 0,
    'retry_delay': timedelta(seconds=5)
}

# Define the DAG using a context manager with updated description and tags.
with DAG(
        dag_id='Lead_scoring_inference_pipeline',
        default_args=default_args,
        description='Lead Scoring Inference Pipeline that processes raw data, validates schemas, loads data, maps features, and generates predictions.',
        schedule_interval='@daily',
        catchup=False,
        tags=['lead_scoring', 'inference', 'pipeline']
) as dag:
    
    # Task: Start MLflow server if not already running.
    start_mlflow = PythonOperator(
        task_id='start_mlflow_server',
        python_callable=start_mlflow_server_if_not_running,
        doc_md="Start the MLflow server if it isn't running already."
    )

    # Task: Check the raw data schema.
    checking_raw_data_schema = PythonOperator(
        task_id='checking_raw_data_schema',
        python_callable=raw_data_schema_check,
        op_args=[CSV_DATA, RAW_DATA_SCHEMA],
        doc_md="Validate the raw CSV data against the defined schema."
    )

    # Task: Load CSV data into the database.
    loading_data = PythonOperator(
        task_id='loading_data',
        python_callable=load_data_into_db,
        op_args=[CSV_DATA],
        doc_md="Load the raw CSV data into the database."
    )

    # Task: Map city tier values.
    mapping_city_tier = PythonOperator(
        task_id='mapping_city_tier',
        python_callable=map_city_tier,
        doc_md="Map city tier values in the data."
    )

    # Task: Map categorical variables.
    mapping_categorical_vars = PythonOperator(
        task_id='mapping_categorical_vars',
        python_callable=map_categorical_vars,
        doc_md="Translate categorical variables into standardized numerical values."
    )

    # Task: Map feature interactions.
    mapping_interactions = PythonOperator(
        task_id='mapping_interactions',
        python_callable=interactions_mapping,
        op_args=[TABLE_NAME],
        doc_md="Generate feature interactions mapping based on the specified table."
    )

    # Task: Clean up interim data in the database.
    remove_interim_data = PythonOperator(
        task_id='cleaning_db',
        python_callable=clean_up_db,
        doc_md="Clean any temporary data in the database post transformation."
    )

    # Task: Check the model input schema.
    checking_model_inputs_schema = PythonOperator(
        task_id='checking_model_inputs_schema',
        python_callable=model_input_schema_check,
        op_args=[TABLE_NAME, MODEL_INPUT_SCHEMA],
        doc_md="Ensure that the transformed data conforms to the model input schema."
    )

    # Task: Encode categorical features.
    encoding_categorical_variables = PythonOperator(
        task_id='encoding_categorical_variables',
        python_callable=encode_features,
        doc_md="Encode categorical variables using appropriate encoding logic."
    )

    # Task: Generate model predictions.
    generating_models_prediction = PythonOperator(
        task_id='generating_models_prediction',
        python_callable=get_models_prediction,
        doc_md="Generate predictions using the deployed ML model."
    )

    # Task: Validate the ratio of model predictions.
    checking_model_prediction_ratio = PythonOperator(
        task_id='checking_model_prediction_ratio',
        python_callable=prediction_ratio_check,
        doc_md="Ensure that the ratio of model predictions is within acceptable bounds."
    )

    # Define task dependencies.
    start_mlflow >> checking_raw_data_schema >> loading_data >> mapping_city_tier \
        >> mapping_categorical_vars >> mapping_interactions >> remove_interim_data \
        >> checking_model_inputs_schema >> encoding_categorical_variables \
        >> generating_models_prediction >> checking_model_prediction_ratio
