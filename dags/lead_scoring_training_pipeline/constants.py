# Training dadatabase
DB_FULL_PATH = '/home/airflow/dags/lead_scoring_data_pipeline/lead_scoring_data_cleaning.db'

# MLflow
MLFLOW_DB = '/home/mlflow/lead_scoring.db'
MLFLOW_TRACKING_URI = "http://0.0.0.0:6006"
MLFLOW_PORT = 6006
MLFLOW_BACKEND_STORE_URI = 'sqlite:////home/mlflow/lead_scoring.db'
MLFLOW_ARTIFACT_ROOT = '/home/mlruns'
EXPERIMENT = "Lead_Scoring_Training_Pipeline"

# model config imported from pycaret experimentation
MODEL_CONFIG = {
    'boosting_type': 'gbdt',
    'learning_rate': 0.01919206058621799,
    'n_estimators': 144,
    'max_depth': -1,
    'num_leaves': 176,
    'min_child_samples': 64,
    'min_child_weight': 0.001,
    'min_split_gain': 0.10013366427103398,
    'reg_alpha': 3.7354322978081316e-10,
    'reg_lambda': 7.804174944814686e-09,
    'feature_fraction': 0.6941874692691037,  # Preferred over colsample_bytree
    'bagging_fraction': 0.7270347957141144,  # Preferred over subsample
    'bagging_freq': 6,                       # Requires matching subsample_freq
    'importance_type': 'split',
    'random_state': 42,
    'n_jobs': 4,
    'silent': 'warn',                        # Deprecated; use verbose=-1 if needed
    'class_weight': None,
    'objective': None,    
    # Redundant/conflicting parameters
    # 'colsample_bytree': 1.0,               # Ignored when feature_fraction is set
    # 'subsample': 1.0,                      # Ignored when bagging_fraction is set
    # 'subsample_freq': 0,                  # Conflicts with bagging_freq > 0
}


# list of the features that need to be there in the final encoded dataframe
ONE_HOT_ENCODED_FEATURES = [
    'total_leads_droppped',
    'referred_lead',
    'city_tier_1.0',
    'city_tier_2.0',
    'city_tier_3.0',
    'first_platform_c_Level0',
    'first_platform_c_Level3',
    'first_platform_c_Level7',
    'first_platform_c_Level1',
    'first_platform_c_Level2',
    'first_platform_c_Level8',
    'first_platform_c_others',
    'first_utm_medium_c_Level0',
    'first_utm_medium_c_Level2',
    'first_utm_medium_c_Level6',
    'first_utm_medium_c_Level3',
    'first_utm_medium_c_Level4',
    'first_utm_medium_c_Level9',
    'first_utm_medium_c_Level11',
    'first_utm_medium_c_Level5',
    'first_utm_medium_c_Level8',
    'first_utm_medium_c_Level20',
    'first_utm_medium_c_Level13',
    'first_utm_medium_c_Level30',
    'first_utm_medium_c_Level33',
    'first_utm_medium_c_Level16',
    'first_utm_medium_c_Level10',
    'first_utm_medium_c_Level15',
    'first_utm_medium_c_Level26',
    'first_utm_medium_c_Level43',
    'first_utm_medium_c_others',
    'first_utm_source_c_Level2',
    'first_utm_source_c_Level0',
    'first_utm_source_c_Level7',
    'first_utm_source_c_Level4',
    'first_utm_source_c_Level6',
    'first_utm_source_c_Level16',
    'first_utm_source_c_Level5',
    'first_utm_source_c_Level14',
    'first_utm_source_c_others',
    'app_complete_flag',
]

# list of features that need to be one-hot encoded
FEATURES_TO_ENCODE = [
    'city_tier',
    'first_platform_c',
    'first_utm_medium_c',
    'first_utm_source_c',
]