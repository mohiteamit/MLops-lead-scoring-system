"""
Module: lead_scoring_data_pipeline
Description: 
    Defines an Airflow DAG that orchestrates the lead scoring data engineering pipeline.
    The pipeline involves creating databases, validating raw data and model input schemas,
    loading data, performing various data transformations and mappings (city tier, categorical features,
    interaction terms) and finally cleaning up the database. The pipeline optimizes the preprocessing
    steps for a robust lead scoring model.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from lead_scoring_data_pipeline.utils import (
    build_dbs,
    load_data_into_db,
    map_city_tier,
    map_categorical_vars,
    interactions_mapping,
    clean_up_db
)
from lead_scoring_data_pipeline.data_validation_checks import (
    raw_data_schema_check,
    model_input_schema_check
)
from lead_scoring_data_pipeline.constants import CSV_DATA, TABLE_NAME
from lead_scoring_data_pipeline.schema import RAW_DATA_SCHEMA, MODEL_INPUT_SCHEMA

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 1, 1),
    'retries': 0,
    'retry_delay': timedelta(seconds=5)
}

# Define the DAG using a context manager (with block)
with DAG(
    dag_id='lead_scoring_data_engineering_pipeline',
    default_args=default_args,
    description='ETL pipeline for preprocessing and validating data for lead scoring model',
    schedule_interval='@daily',
    catchup=False,
    tags=['lead_scoring', 'ETL', 'data_pipeline', 'data_validation']
) as ML_data_cleaning_dag:

    # Task to create necessary databases
    building_db = PythonOperator(
        task_id='building_db',
        python_callable=build_dbs,
        doc_md="""\
        Build Databases: Creates required database structures for the pipeline.
        Essential for storing and querying preprocessed lead scoring data.
        """
    )

    # Task to check raw data schema
    checking_raw_data_schema = PythonOperator(
        task_id='checking_raw_data_schema',
        python_callable=raw_data_schema_check,
        op_args=[CSV_DATA, RAW_DATA_SCHEMA],
        doc_md="""\
        Raw Data Schema Check: Validates that the incoming CSV data conforms to the expected schema.
        Prevents schema-related issues downstream.
        """
    )

    # Task to load data into database
    loading_data = PythonOperator(
        task_id='loading_data',
        python_callable=load_data_into_db,
        op_args=[CSV_DATA],
        doc_md="""\
        Data Loading: Ingests CSV data into the database for subsequent transformations.
        """
    )

    # Task to map city tier
    mapping_city_tier = PythonOperator(
        task_id='mapping_city_tier',
        python_callable=map_city_tier,
        doc_md="""\
        City Tier Mapping: Transforms city-related information into standardized tier classifications.
        Improves data consistency for location-based analytics.
        """
    )

    # Task to map categorical variables
    mapping_categorical_vars = PythonOperator(
        task_id='mapping_categorical_vars',
        python_callable=map_categorical_vars,
        doc_md="""\
        Categorical Variables Mapping: Processes and encodes categorical features into numerical representations.
        Essential for machine learning model compatibility.
        """
    )

    # Task to map interactions
    mapping_interactions = PythonOperator(
        task_id='mapping_interactions',
        python_callable=interactions_mapping,
        op_args=[TABLE_NAME],
        doc_md="""\
        Interaction Mapping: Computes interaction terms between features, enhancing the feature space for modeling.
        """
    )

    # Task to check model input schema
    checking_model_inputs_schema = PythonOperator(
        task_id='checking_model_inputs_schema',
        python_callable=model_input_schema_check,
        op_args=[TABLE_NAME, MODEL_INPUT_SCHEMA],
        doc_md="""\
        Model Input Schema Check: Ensures that the transformed data matches the expected schema for the lead scoring model.
        """
    )

    # Task to clean up the database
    clean_up_db = PythonOperator(
        task_id='cleaning_db',
        python_callable=clean_up_db,
        doc_md="""\
        Database Cleanup: Performs post-processing cleanup operations to maintain optimal database conditions.
        """
    )

    # Define task dependencies
    building_db >> checking_raw_data_schema >> loading_data >> mapping_city_tier >> \
        mapping_categorical_vars >> mapping_interactions >> checking_model_inputs_schema >> clean_up_db
