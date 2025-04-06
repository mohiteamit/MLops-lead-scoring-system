# TODO - DOES NOT WORK. SQLITE DB LOCKS.

from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 1, 1),
    'retries': 0,
}

with DAG(
    dag_id='master_pipeline_dag',
    default_args=default_args,
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['master', 'unit tests']
) as dag:

    # Trigger unit test DAG
    run_unit_tests = TriggerDagRunOperator(
        task_id="run_unit_test_dag",
        trigger_dag_id="unit_test_runner",
        wait_for_completion=False,
        reset_dag_run=True
    )

    # Wait for unit test to complete — must match same execution_date
    wait_for_unit_test = ExternalTaskSensor(
        task_id='wait_for_unit_test',
        external_dag_id='unit_test_runner',
        external_task_id=None,  # Wait for whole DAG
        execution_delta=timedelta(0),  # Force same execution_date matching
        mode='poke',
        timeout=60,
        poke_interval=30
    )

    # Trigger next DAGs
    run_data_pipeline = TriggerDagRunOperator(
        task_id="run_data_pipeline_dag",
        trigger_dag_id="Lead_Scoring_Data_Engineering_Pipeline",
        wait_for_completion=False,
        reset_dag_run=True
    )

    wait_for_data_pipeline = ExternalTaskSensor(
        task_id='wait_for_data_pipeline',
        external_dag_id='Lead_Scoring_Data_Engineering_Pipeline',
        external_task_id=None,
        execution_delta=timedelta(0),
        mode='poke',
        timeout=60,
        poke_interval=30
    )

    run_training_pipeline = TriggerDagRunOperator(
        task_id="run_training_pipeline_dag",
        trigger_dag_id="Lead_scoring_training_pipeline",
        wait_for_completion=False,
        reset_dag_run=True
    )

    wait_for_training_pipeline = ExternalTaskSensor(
        task_id='wait_for_training_pipeline',
        external_dag_id='Lead_scoring_training_pipeline',
        external_task_id=None,
        execution_delta=timedelta(0),
        mode='poke',
        timeout=60,
        poke_interval=30
    )

    run_inference_pipeline = TriggerDagRunOperator(
        task_id="run_inference_pipeline_dag",
        trigger_dag_id="Lead_scoring_inference_pipeline",
        wait_for_completion=False,
        reset_dag_run=True
    )

    (
        run_unit_tests
        >> wait_for_unit_test
        >> run_data_pipeline
        >> wait_for_data_pipeline
        >> run_training_pipeline
        >> wait_for_training_pipeline
        >> run_inference_pipeline
    )
