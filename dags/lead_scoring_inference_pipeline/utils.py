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

# def start_mlflow_server_if_not_running():
#     """
#     Checks if MLflow is running on the given port. If not, attempts to start it.
    
#     Args:
#         port (int): Port number to check/start MLflow on.
#         backend_store_uri (str): Path to the SQLite backend store URI.
#         artifact_root (str): Directory to store MLflow artifacts.
    
#     Raises:
#         RuntimeError: If MLflow fails to start.
#     """
#     def is_port_in_use(port):
#         """Check if a port is already being used."""
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#             return s.connect_ex(('0.0.0.0', port)) == 0 or s.connect_ex(('127.0.0.1', port)) == 0

#     if not is_port_in_use(MLFLOW_PORT):
#         try:
#             print("MLflow is not running. Attempting to start it...")
#             subprocess.Popen([
#                 "mlflow", "server",
#                 "--backend-store-uri", MLFLOW_BACKEND_STORE_URI,
#                 "--default-artifact-root", MLFLOW_ARTIFACT_ROOT,
#                 "--port", str(MLFLOW_PORT),
#                 "--host", "0.0.0.0"
#             ])
#             time.sleep(5)  # Give server time to start
#             if not is_port_in_use(MLFLOW_PORT):
#                 raise RuntimeError("Failed to start MLflow server.")
#             print(f"MLflow server started on port {MLFLOW_PORT}.")
#         except Exception as e:
#             print(f"Error starting MLflow: {e}")
#             raise e
#     else:
#         print(f"MLflow server already running on port {MLFLOW_PORT}.")

# def encode_features():
#     """
#     One-hot encodes the categorical features in the training dataset.
#     Encoded features and the target variable are saved in separate tables.

#     INPUTS:
#         DB_FULL_PATH             : Complete path of the database file.
#         ONE_HOT_ENCODED_FEATURES : List of features for the final encoded DataFrame.
#         FEATURES_TO_ENCODE       : List of features from cleaned data to be one-hot encoded.

#     OUTPUT:
#         - Encoded features saved in the 'INFERENCE' table.
#         - (Target saving commented out.)

#     USAGE:
#         encode_features()

#     NOTE:
#         Used in Airflow pipelines. Print statements are included for non-Airflow testing.
#     """
#     connection = None
#     try:
#         connection = sqlite3.connect(DB_FULL_PATH)
#         model_input_df = pd.read_sql('SELECT * FROM MODEL_INPUT', connection)

#         # Prepare empty DataFrames for final encoded features and intermediate one-hot encoded data.
#         final_encoded_df = pd.DataFrame(columns=ONE_HOT_ENCODED_FEATURES)
#         encoded_dummies_df = pd.DataFrame()

#         # Perform one-hot encoding for each feature in FEATURES_TO_ENCODE.
#         for feature in FEATURES_TO_ENCODE:
#             if feature in model_input_df.columns:
#                 dummies = pd.get_dummies(model_input_df[feature]).add_prefix(f"{feature}_")
#                 encoded_dummies_df = pd.concat([encoded_dummies_df, dummies], axis=1)
#             else:
#                 error_msg = f"Feature '{feature}' not found in MODEL_INPUT dataframe."
#                 print(error_msg)
#                 connection.close()
#                 raise AirflowException(error_msg)

#         # Combine original and encoded columns into the final DataFrame.
#         for col in final_encoded_df.columns:
#             if col in model_input_df.columns:
#                 final_encoded_df[col] = model_input_df[col]
#             if col in encoded_dummies_df.columns:
#                 final_encoded_df[col] = encoded_dummies_df[col]

#         final_encoded_df.fillna(0, inplace=True)

#         # Save features (excluding target) in a table.
#         if 'app_complete_flag' in final_encoded_df.columns:
#             features_to_save_df = final_encoded_df.drop(['app_complete_flag'], axis=1)
#         else:
#             features_to_save_df = final_encoded_df.copy()
#         features_to_save_df.to_sql(name='INFERENCE', con=connection, if_exists='replace', index=False)
#         connection.close()
#         print("encode_features executed successfully.")
#     except Exception as e:
#         if connection:
#             connection.close()
#         print(f"Error in encode_features: {e}")
#         raise AirflowException(e)

# def input_features_check():
#     """
#     Checks whether all required input columns are present in the inference data.
#     Logs the status to the console and raises an exception if columns mismatch,
#     ensuring the pipeline doesn't break silently due to missing columns.

#     INPUTS:
#         - Database file name and path (via DB_FULL_PATH).
#         - ONE_HOT_ENCODED_FEATURES: List of expected input features.

#     OUTPUT:
#         Logs the presence of all input columns.
#           - 'All the model input are present' if complete.
#           - Raises ValueError if there is a mismatch.

#     USAGE:
#         input_features_check()
#     """
#     connection = sqlite3.connect(os.path.join(DB_FULL_PATH))
#     inference_df = pd.read_sql('SELECT * FROM INFERENCE', connection)

#     expected_columns = set(ONE_HOT_ENCODED_FEATURES)
#     actual_columns = set(inference_df.columns)

#     missing_columns = expected_columns - actual_columns
#     extra_columns = actual_columns - expected_columns

#     if missing_columns or extra_columns:
#         print('Some of the model inputs are missing or extra.')
#         print('Missing columns:', missing_columns)
#         print('Extra columns:', extra_columns)
#         connection.close()
#         raise ValueError(f"Mismatch in input features. Missing: {missing_columns}, Extra: {extra_columns}")
#     else:
#         print('All the model input are present')
#     connection.close()

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
        inference_df = pd.read_sql('SELECT * FROM INFERENCE', connection)

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
