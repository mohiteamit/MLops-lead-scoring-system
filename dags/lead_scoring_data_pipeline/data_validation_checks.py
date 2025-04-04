from lead_scoring_data_pipeline.constants import DB_FULL_PATH
import pandas as pd
import sqlite3

def raw_data_schema_check(csv_file_path : str, schema : list):
    '''
    This function verifies that all the columns mentioned in RAW_DATA_SCHEMA 
    are present in the 'leadscoring.csv' file. Raises a ValueError with a detailed 
    message if the schema does not match or if file loading fails.
    '''
    try:
        df_lead_data = pd.read_csv(csv_file_path, index_col=[0])
    except Exception as e:
        raise Exception(f"Failed to read CSV file at {csv_file_path}: {e}")
    
    if set(df_lead_data.columns) == set(schema):
        print('Raw datas schema is in line with the schema present in schema.py')
    else:
        missing = set(schema) - set(df_lead_data.columns)
        extra = set(df_lead_data.columns) - set(schema)
        raise ValueError(f"Raw data schema mismatch: Missing columns {missing}, Unexpected columns {extra}. Please verify the CSV file and schema definitions.")

def model_input_schema_check(table_name, schema : list):
    '''
    This function verifies that all the columns mentioned in MODEL_INPUT_SCHEMA 
    are present in the 'model_input' table in the database. Raises a ValueError with a detailed 
    message if the schema does not match or if connection/query fails.
    '''
    try:
        conn = sqlite3.connect(DB_FULL_PATH)
    except Exception as e:
        raise Exception(f"Failed to connect to the database at {DB_FULL_PATH}: {e}")
    
    try:
        df_model_in = pd.read_sql(f'SELECT * FROM {table_name}', conn)
    except Exception as e:
        raise Exception(f"Failed to read table '{table_name}' from the database at {DB_FULL_PATH}: {e}")
    finally:
        conn.close()
    
    if set(df_model_in.columns) == set(schema):
        print('Models input schema is in line with the schema present in schema.py')
    else:
        missing = set(schema) - set(df_model_in.columns)
        extra = set(df_model_in.columns) - set(schema)
        raise ValueError(f"Model input schema mismatch: Missing columns {missing}, Unexpected columns {extra}. Please verify the database table and schema definitions.")
