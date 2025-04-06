import os
import socket
import subprocess
import time
import sqlite3
from datetime import datetime

import mlflow
import pandas as pd
from airflow.exceptions import AirflowException

from lead_scoring_inference_pipeline.constants import (
    DB_FULL_PATH,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    STAGE,
    PREDICTION_DIST_TXT
)

def encode_features():
    # common encode_features function for train and inference
    pass

def get_models_prediction():
    """
    Loads the production model from the MLflow registry and generates predictions
    on new data. The predictions are then stored alongside the input data.

    INPUTS:
        - Database file name and path (via DB_FULL_PATH).
        - Model name and stage (from constants).

    OUTPUT:
        Stores predicted values in the 'PREDICTIONS' table.

    USAGE:
        get_models_prediction()
    """
    connection = None
    try:
        # Set MLflow tracking URI and load the latest production model.
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{STAGE}")

        # Read inference data from the database.
        connection = sqlite3.connect(os.path.join(DB_FULL_PATH))
        inference_df = pd.read_sql('SELECT * FROM FEATURES', connection)

        # Generate predictions and store in DataFrame.
        predictions = loaded_model.predict(inference_df)
        inference_df['pred_app_complete_flag'] = predictions

        # Save predictions in the database.
        inference_df.to_sql(name='PREDICTIONS', con=connection, if_exists='replace', index=False)
        connection.close()
        print("get_models_prediction executed successfully.")
    except Exception as e:
        if connection:
            connection.close()
        print(f"Error in get_models_prediction: {e}")
        raise AirflowException(e)

def prediction_ratio_check():
    """
    Calculates the distribution (percentage) of predicted values (1s and 0s) and
    writes the result with a timestamp to the prediction_distribution.txt file.
    This monitoring helps detect any drift in model predictions.

    INPUTS:
        - Database file name and path (via DB_FULL_PATH).

    OUTPUT:
        Appends the distribution output with timestamp to PREDICTION_DIST_TXT.

    USAGE:
        prediction_ratio_check()
    """
    connection = None
    try:
        connection = sqlite3.connect(os.path.join(DB_FULL_PATH))
        predictions_df = pd.read_sql('SELECT * FROM PREDICTIONS', connection)

        # Calculate normalized value counts for the prediction column.
        value_counts = predictions_df['pred_app_complete_flag'].value_counts(normalize=True)

        # Create log entry with timestamp and percentages.
        current_time = datetime.now()
        log_entry = (
            f"{str(current_time)} % of 1 = {str(value_counts.get(1, 0))} "
            f"% of 0 = {str(value_counts.get(0, 0))}"
        )
        with open(PREDICTION_DIST_TXT, 'a') as file:
            file.write(log_entry + "\n")

        connection.close()
        print("prediction_ratio_check executed successfully.")
    except Exception as e:
        if connection:
            connection.close()
        print(f"Error in prediction_ratio_check: {e}")
        raise AirflowException(e)

def input_features_check():
    # Using checking pipeline as training pipeline which is robust.
    pass