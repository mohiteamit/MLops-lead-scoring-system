import socket
import subprocess
import time
import pandas as pd
import numpy as np
import sqlite3
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb
from lead_scoring_training_pipeline.constants import DB_FULL_PATH, ONE_HOT_ENCODED_FEATURES, FEATURES_TO_ENCODE, MLFLOW_DB, MODEL_CONFIG, MLFLOW_TRACKING_URI, EXPERIMENT
from lead_scoring_training_pipeline.constants import MLFLOW_PORT, MLFLOW_BACKEND_STORE_URI, MLFLOW_ARTIFACT_ROOT

def start_mlflow_server_if_not_running():
    """
    Checks if MLflow is running on the given port. If not, attempts to start it.
    
    Args:
        port (int): Port number to check/start MLflow on.
        backend_store_uri (str): Path to the SQLite backend store URI.
        artifact_root (str): Directory to store MLflow artifacts.
    
    Raises:
        RuntimeError: If MLflow fails to start.
    """
    def is_port_in_use(port):
        """Check if a port is already being used."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('0.0.0.0', port)) == 0 or s.connect_ex(('127.0.0.1', port)) == 0

    if not is_port_in_use(MLFLOW_PORT):
        try:
            print("MLflow is not running. Attempting to start it...")
            subprocess.Popen([
                "mlflow", "server",
                "--backend-store-uri", MLFLOW_BACKEND_STORE_URI,
                "--default-artifact-root", MLFLOW_ARTIFACT_ROOT,
                "--port", str(MLFLOW_PORT),
                "--host", "0.0.0.0"
            ])
            time.sleep(5)  # Give server time to start
            if not is_port_in_use(MLFLOW_PORT):
                raise RuntimeError("Failed to start MLflow server.")
            print(f"MLflow server started on port {MLFLOW_PORT}.")
        except Exception as e:
            print(f"Error starting MLflow: {e}")
            raise e
    else:
        print(f"MLflow server already running on port {MLFLOW_PORT}.")

def encode_features():
    """
    One-hot encodes the categorical features present in our training dataset.
    This encoding is needed for feeding categorical data to many scikit-learn models.

    INPUTS
        DB_FULL_PATH                  : Complete path of the database file.
        ONE_HOT_ENCODED_FEATURES      : List of features that should be in the final encoded dataframe.
        FEATURES_TO_ENCODE            : List of features from cleaned data that need to be one-hot encoded.
       
    OUTPUT
        1. Saves the encoded features in a table named 'FEATURES'.
        2. Saves the target variable in a separate table named 'TARGET'.

    SAMPLE USAGE
        encode_features()

    NOTE:
        Function used in Airflow pipelines. During non-Airflow testing,
        print statements can help track the progress.
    """
    conn = sqlite3.connect(DB_FULL_PATH)
    input_data_df = pd.read_sql('SELECT * FROM model_input', conn)

    # Create empty DataFrame for encoded data and a placeholder for intermediate data.
    encoded_features_df = pd.DataFrame(columns=ONE_HOT_ENCODED_FEATURES)
    intermediate_encoded_df = pd.DataFrame()

    # Encode the features using get_dummies()
    for feature in FEATURES_TO_ENCODE:
        if feature in input_data_df.columns:
            encoded = pd.get_dummies(input_data_df[feature])
            encoded = encoded.add_prefix(feature + '_')
            intermediate_encoded_df = pd.concat([intermediate_encoded_df, encoded], axis=1)
        else:
            msg = f"Feature '{feature}' not found in model_input dataframe."
            print(msg)
            conn.close()
            raise ValueError(msg)

    # Combine the encoded features into a single dataframe
    for col in encoded_features_df.columns:
        if col in input_data_df.columns:
            encoded_features_df[col] = input_data_df[col]
        if col in intermediate_encoded_df.columns:
            encoded_features_df[col] = intermediate_encoded_df[col]

    encoded_features_df.fillna(0, inplace=True)

    # Save the features and target in separate tables
    if 'app_complete_flag' not in encoded_features_df.columns:
        msg = "Target feature 'app_complete_flag' not found in dataframe."
        print(msg)
        conn.close()
        raise ValueError(msg)

    features_to_save_df = encoded_features_df.drop(['app_complete_flag'], axis=1)
    target_to_save_df = encoded_features_df[['app_complete_flag']]
    features_to_save_df.to_sql(name='FEATURES', con=conn, if_exists='replace', index=False)
    target_to_save_df.to_sql(name='TARGET', con=conn, if_exists='replace', index=False)

    conn.close()

def get_trained_model():
    """
    - Fetches the best run by AUC from the MLflow SQLite DB.
    - Sets tracking URI and experiment.
    - Trains a new LightGBM model.
    - Logs model, parameters, and AUC to MLflow.

    In an Airflow pipeline, exceptions are raised for error handling.
    Print statements are kept for testing outside of Airflow.
    """

    # Step 1: Get best run info from MLflow DB
    conn = sqlite3.connect(f"file:{MLFLOW_DB}?mode=ro", uri=True)
    best_run_metrics_query = """
        SELECT run_uuid, value
        FROM metrics
        WHERE key = 'AUC'
        ORDER BY value DESC
        LIMIT 1
    """
    best_run_metrics_df = pd.read_sql(best_run_metrics_query, conn)

    if best_run_metrics_df.empty:
        msg = "No runs found with the specified metric 'AUC'."
        print(msg)
        conn.close()
        raise ValueError(msg)

    best_run_id = best_run_metrics_df.loc[0, 'run_uuid']
    best_auc = best_run_metrics_df.loc[0, 'value']

    model_name_query = f"""
        SELECT value
        FROM tags
        WHERE run_uuid = '{best_run_id}' AND key = 'mlflow.runName'
    """
    run_tags_df = pd.read_sql(model_name_query, conn)
    conn.close()

    model_name = run_tags_df['value'].iloc[0] if not run_tags_df.empty else "Unknown Model"

    print(f"Best run ID: {best_run_id}")
    print(f"Best AUC Score: {best_auc}")
    print(f"Model Name: {model_name}")

    # Step 2: Train and log a new model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    conn = sqlite3.connect(DB_FULL_PATH)
    X = pd.read_sql('SELECT * FROM FEATURES', conn)
    y = pd.read_sql('SELECT * FROM TARGET', conn)
    conn.close()

    with mlflow.start_run(run_name='Training_LIGHTGBM') as mlrun:
        model = lgb.LGBMClassifier(**MODEL_CONFIG)
        scores = cross_val_score(model, X, y.values.ravel(), scoring='roc_auc', cv=5)
        mean_auc = np.mean(scores)
        mlflow.log_metric("AUC", mean_auc)
        print(f"New model logged with cross-validated AUC: {mean_auc:.4f}")

    # # Train/test split
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.9, random_state=0
    # )

    # with mlflow.start_run(run_name='run_LightGB') as mlrun:
    #     model = lgb.LGBMClassifier()
    #     model.set_params(**MODEL_CONFIG)
    #     model.fit(X_train, y_train)

    #     # Log model
    #     mlflow.sklearn.log_model(
    #         sk_model=model,
    #         artifact_path="models",
    #         registered_model_name='LightGBM'
    #     )

    #     # Log parameters
    #     mlflow.log_params(MODEL_CONFIG)

    #     # Predict and log AUC
    #     y_pred = model.predict(X_test)
    #     auc = roc_auc_score(y_test, y_pred)
    #     mlflow.log_metric('AUC', auc)

    #     print(f"✅ New model trained and logged with AUC: {auc:.4f}")
