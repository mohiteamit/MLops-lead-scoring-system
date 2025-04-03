"""
Import necessary modules
############################################################################## 
"""
import os
from lead_scoring_data_pipeline.constants import DATA_DIRECTORY, CSV_FILE_NAME, DB_PATH, DB_FILE_NAME
from lead_scoring_data_pipeline.schema import RAW_DATA_SCHEMA, MODEL_INPUT_SCHEMA

import pandas as pd
import sqlite3

###############################################################################
# Define function to validate raw data's schema
###############################################################################

def raw_data_schema_check():
    '''
    This function verifies that all the columns mentioned in RAW_DATA_SCHEMA 
    are present in the 'leadscoring.csv' file. Raises a ValueError with a detailed 
    message if the schema does not match or if file loading fails.
    '''
    file_path = os.path.join(DATA_DIRECTORY, CSV_FILE_NAME)
    try:
        df_lead_data = pd.read_csv(file_path, index_col=[0])
    except Exception as e:
        raise Exception(f"Failed to read CSV file at {file_path}: {e}")
    
    if set(df_lead_data.columns) == set(RAW_DATA_SCHEMA):
        print('Raw datas schema is in line with the schema present in schema.py')
    else:
        missing = set(RAW_DATA_SCHEMA) - set(df_lead_data.columns)
        extra = set(df_lead_data.columns) - set(RAW_DATA_SCHEMA)
        raise ValueError(f"Raw data schema mismatch: Missing columns {missing}, Unexpected columns {extra}. Please verify the CSV file and schema definitions.")

###############################################################################
# Define function to validate model's input schema
###############################################################################

def model_input_schema_check():
    '''
    This function verifies that all the columns mentioned in MODEL_INPUT_SCHEMA 
    are present in the 'model_input' table in the database. Raises a ValueError with a detailed 
    message if the schema does not match or if connection/query fails.
    '''
    db_file_path = os.path.join(DB_PATH, DB_FILE_NAME)
    try:
        conn = sqlite3.connect(db_file_path)
    except Exception as e:
        raise Exception(f"Failed to connect to the database at {db_file_path}: {e}")
    
    try:
        df_model_in = pd.read_sql('select * from model_input', conn)
    except Exception as e:
        raise Exception(f"Failed to read table 'model_input' from the database at {db_file_path}: {e}")
    finally:
        conn.close()
    
    if set(df_model_in.columns) == set(MODEL_INPUT_SCHEMA):
        print('Models input schema is in line with the schema present in schema.py')
    else:
        missing = set(MODEL_INPUT_SCHEMA) - set(df_model_in.columns)
        extra = set(df_model_in.columns) - set(MODEL_INPUT_SCHEMA)
        raise ValueError(f"Model input schema mismatch: Missing columns {missing}, Unexpected columns {extra}. Please verify the database table and schema definitions.")
