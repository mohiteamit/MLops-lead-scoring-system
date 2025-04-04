# DATA STORE
DB_FULL_PATH = '/home/airflow/dags/lead_scoring_data_pipeline/lead_scoring_data_cleaning.db'
TABLE_NAME = 'MODEL_INPUT'
TABLE_NAME = 'TRAINING'

# INPUT DATA
CSV_DATA = '/home/airflow/dags/lead_scoring_data_pipeline/data/leadscoring.csv'

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
