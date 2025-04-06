from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from lead_scoring_data_pipeline.utils import (
    load_data_into_db, map_city_tier, map_categorical_vars, interactions_mapping, clean_up_db)
from lead_scoring_data_pipeline.data_validation_checks import (
    raw_data_schema_check, model_input_schema_check)
from lead_scoring_training_pipeline.utils import (
    start_mlflow_server_if_not_running, encode_features)
from lead_scoring_inference_pipeline.utils import (
    get_models_prediction, prediction_ratio_check)
from lead_scoring_inference_pipeline.schema import (
    RAW_DATA_SCHEMA, MODEL_INPUT_SCHEMA)
from lead_scoring_inference_pipeline.constants import (CSV_DATA, TABLE_NAME)

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 9, 14),
    'retries': 1,
    'retry_delay': timedelta(seconds=5)
}


Lead_scoring_inference_dag = DAG(
    dag_id='Lead_scoring_inference_pipeline',
    default_args=default_args,
    description='Inference pipeline of Lead Scoring system',
    schedule_interval='@daily',
    catchup=False
)

start_mlflow = PythonOperator(
    task_id='start_mlflow_server',
    python_callable=start_mlflow_server_if_not_running,
    dag=Lead_scoring_inference_dag
)

checking_raw_data_schema = PythonOperator(
    task_id='checking_raw_data_schema',
    python_callable=raw_data_schema_check,
    op_args=[CSV_DATA, RAW_DATA_SCHEMA],
    dag=Lead_scoring_inference_dag)

loading_data = PythonOperator(
    task_id='loading_data',
    python_callable=load_data_into_db,
    op_args=[CSV_DATA],
    dag=Lead_scoring_inference_dag)

mapping_city_tier = PythonOperator(
    task_id='mapping_city_tier',
    python_callable=map_city_tier,
    dag=Lead_scoring_inference_dag)

mapping_categorical_vars = PythonOperator(
    task_id='mapping_categorical_vars',
    python_callable=map_categorical_vars,
    dag=Lead_scoring_inference_dag)

mapping_interactions = PythonOperator(
    task_id='mapping_interactions',
    python_callable=interactions_mapping,
    op_args=[TABLE_NAME],
    dag=Lead_scoring_inference_dag)

remove_interim_data = PythonOperator(
    task_id='cleaning_db',
    python_callable=clean_up_db,
    dag=Lead_scoring_inference_dag)

checking_model_inputs_schema = PythonOperator(
    task_id='checking_model_inputs_schema',
    python_callable=model_input_schema_check,
    op_args=[TABLE_NAME, MODEL_INPUT_SCHEMA],
    dag=Lead_scoring_inference_dag)


encoding_categorical_variables = PythonOperator(
    task_id='encoding_categorical_variables',
    python_callable=encode_features,
    dag=Lead_scoring_inference_dag)

# checking_input_features = PythonOperator(
#         task_id = 'checking_input_features',
#         python_callable = model_input_schema_check,
#         dag = Lead_scoring_inference_dag)

generating_models_prediction = PythonOperator(
    task_id='generating_models_prediction',
    python_callable=get_models_prediction,
    dag=Lead_scoring_inference_dag)


checking_model_prediction_ratio = PythonOperator(
    task_id='checking_model_prediction_ratio',
    python_callable=prediction_ratio_check,
    dag=Lead_scoring_inference_dag)

start_mlflow >> checking_raw_data_schema >> loading_data >> mapping_city_tier >> mapping_categorical_vars >> mapping_interactions >> remove_interim_data >> checking_model_inputs_schema >> encoding_categorical_variables >> generating_models_prediction >> checking_model_prediction_ratio
