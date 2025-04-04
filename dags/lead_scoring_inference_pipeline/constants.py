# DATA STORE
DB_FULL_PATH = '/home/airflow/dags/lead_scoring_data_pipeline/lead_scoring_data_cleaning.db'
TABLE_NAME = 'INFERENCE'

# INPUT DATA
CSV_DATA = '/home/airflow/dags/lead_scoring_data_pipeline/data/leadscoring_inference.csv'

INTERACTION_MAPPING = "/home/airflow/dags/lead_scoring_data_pipeline/mappings/interaction_mapping.csv"

INDEX_COLUMNS_TRAINING = [
    'created_date', 
    'city_tier', 
    'first_platform_c',
    'first_utm_medium_c', 
    'first_utm_source_c', 
    'total_leads_droppped',
    'referred_lead', 
    'app_complete_flag'
]

INDEX_COLUMNS_INFERENCE = [
    'created_date', 
    'city_tier', 
    'first_platform_c',
    'first_utm_medium_c', 
    'first_utm_source_c',
    'total_leads_droppped',
    'referred_lead'
]

NOT_FEATURES = [
    'created_date', 
    'assistance_interaction', 
    'career_interaction',
    'payment_interaction', 
    'social_interaction', 
    'syllabus_interaction'
]

# MLflow
MLFLOW_DB = '/home/mlflow/lead_scoring.db'
MLFLOW_TRACKING_URI = "http://0.0.0.0:6006"
MLFLOW_PORT = 6006
MLFLOW_BACKEND_STORE_URI = 'sqlite:////home/mlflow/lead_scoring.db'
MLFLOW_ARTIFACT_ROOT = '/home/mlruns'
EXPERIMENT = "Lead_Scoring_Training_Pipeline"

# experiment, model name and stage to load the model from mlflow model registry
MODEL_NAME = "LightGBM"
STAGE = "Production"

# Inference predictions
PREDICTION_DIST_TXT = '/home/airflow/dags/lead_scoring_inference_pipeline/prediction_distribution.txt'
