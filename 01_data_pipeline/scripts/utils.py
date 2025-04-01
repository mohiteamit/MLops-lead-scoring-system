"""
Utility functions for building the database and loading/mapping data.

This module contains functions:
    - build_dbs: Checks and creates the SQLite database if missing.
    - load_data_into_db: Loads CSV data into the database.
    - map_city_tier: Maps cities to tiers.
    - map_categorical_vars: Maps insignificant categorical variables to "others".
    - interactions_mapping: Maps interaction columns for training/inference.
"""

##############################################################################
# Import necessary modules and constants
##############################################################################
import os
import sqlite3
from sqlite3 import Error
import pandas as pd

from constants import *
from city_tier_mapping import city_tier_mapping
from significant_categorical_level import list_platform, list_medium, list_source


##############################################################################
# Database Building Function
##############################################################################
def build_dbs():
    """
    Checks if the db file exists in DB_PATH with DB_FILE_NAME. If it does not,
    creates the SQLite database file.

    Returns:
         'DB Exists' if the file already exists.
         'DB created' if a new database is created.
         Error message string if database creation fails.
    """
    db_full_path = os.path.join(DB_PATH, DB_FILE_NAME)

    if os.path.exists(db_full_path):
        print('DB Already Exists')
        return 'DB Exists'
    else:
        print('Creating Database')
        try:
            conn = sqlite3.connect(db_full_path)
            if conn:
                conn.close()
                print('New DB Created')
                return 'DB created'
        except Error as e:
            print(f"Error creating database: {e}")
            return f"DB creation failed: {e}"

##############################################################################
# Data Loading Function
##############################################################################
def load_data_into_db():
    """
    Loads the CSV data from DATA_DIRECTORY into the sqlite db and replaces
    missing values in 'total_leads_droppped' and 'referred_lead' with 0.

    The data is stored in a table named 'loaded_data' (overwrites if exists).
    """
    connection = sqlite3.connect(os.path.join(DB_PATH, DB_FILE_NAME))
    df_lead_scoring = pd.read_csv(os.path.join(DATA_DIRECTORY, 'leadscoring_inference.csv'))

    df_lead_scoring['total_leads_droppped'] = df_lead_scoring['total_leads_droppped'].fillna(0)
    df_lead_scoring['referred_lead'] = df_lead_scoring['referred_lead'].fillna(0)

    df_lead_scoring.to_sql(name='loaded_data', con=connection, if_exists='replace', index=False)
    connection.close()

##############################################################################
# City Tier Mapping Function
##############################################################################
def map_city_tier():
    """
    Maps cities from the 'city_mapped' column to their tiers using the
    city_tier_mapping dictionary. Unmapped cities are assigned a tier of 3.0.
    The resulting dataframe (without the original 'city_mapped' column) is
    stored in the table 'city_tier_mapped'.
    """
    connection = sqlite3.connect(os.path.join(DB_PATH, DB_FILE_NAME))
    df_lead_scoring = pd.read_sql('SELECT * FROM loaded_data', connection)

    df_lead_scoring["city_tier"] = df_lead_scoring["city_mapped"].map(city_tier_mapping)
    df_lead_scoring["city_tier"] = df_lead_scoring["city_tier"].fillna(3.0)
    df_lead_scoring = df_lead_scoring.drop(['city_mapped'], axis=1)

    df_lead_scoring.to_sql(name='city_tier_mapped', con=connection, if_exists='replace', index=False)
    connection.close()

##############################################################################
# Categorical Variable Mapping Function
##############################################################################
def map_categorical_vars():
    """
    Maps insignificant values in 'first_platform_c', 'first_utm_medium_c', and
    'first_utm_source_c' to "others" if they are not in the respective significant
    levels lists (list_platform, list_medium, list_source). The final dataframe
    is stored in the table 'categorical_variables_mapped'.
    """
    connection = sqlite3.connect(os.path.join(DB_PATH, DB_FILE_NAME))
    df_lead_scoring = pd.read_sql('SELECT * FROM city_tier_mapped', connection)

    # Process first_platform_c mapping
    new_df = df_lead_scoring[~df_lead_scoring['first_platform_c'].isin(list_platform)]
    new_df['first_platform_c'] = "others"
    old_df = df_lead_scoring[df_lead_scoring['first_platform_c'].isin(list_platform)]
    df = pd.concat([new_df, old_df])

    # Process first_utm_medium_c mapping
    new_df = df[~df['first_utm_medium_c'].isin(list_medium)]
    new_df['first_utm_medium_c'] = "others"
    old_df = df[df['first_utm_medium_c'].isin(list_medium)]
    df = pd.concat([new_df, old_df])

    # Process first_utm_source_c mapping
    new_df = df[~df['first_utm_source_c'].isin(list_source)]
    new_df['first_utm_source_c'] = "others"
    old_df = df[df['first_utm_source_c'].isin(list_source)]
    df = pd.concat([new_df, old_df])

    df = df.drop_duplicates()
    df.to_sql(name='categorical_variables_mapped', con=connection, if_exists='replace', index=False)
    connection.close()

##############################################################################
# Interaction Mapping Function
##############################################################################
def interactions_mapping():
    """
    Maps the interaction columns into 4 unique interaction groups using
    mappings from INTERACTION_MAPPING (CSV file). Depending on whether the
    'app_complete_flag' column is present, it selects appropriate index columns.
    The transformed data is saved in two tables:
        - 'interactions_mapped'
        - 'model_input' (after dropping the features in NOT_FEATURES)
    """
    connection = sqlite3.connect(os.path.join(DB_PATH, DB_FILE_NAME))
    df = pd.read_sql('SELECT * FROM categorical_variables_mapped', connection)
    
    if 'app_complete_flag' in df.columns:
        index_variable = INDEX_COLUMNS_TRAINING
    else:
        index_variable = INDEX_COLUMNS_INFERENCE
        
    df_event_mapping = pd.read_csv(INTERACTION_MAPPING, index_col=[0])
    
    df_unpivot = pd.melt(df, id_vars=index_variable, var_name='interaction_type', value_name='interaction_value')
    df_unpivot['interaction_value'] = df_unpivot['interaction_value'].fillna(0)
    
    df = pd.merge(df_unpivot, df_event_mapping, on='interaction_type', how='left')
    df = df.drop(['interaction_type'], axis=1)
    
    df_pivot = df.pivot_table(values='interaction_value', index=index_variable,
                              columns='interaction_mapping', aggfunc='sum').reset_index()
    
    df_pivot.to_sql(name='interactions_mapped', con=connection, if_exists='replace', index=False)
    
    df_model_input = df_pivot.drop(NOT_FEATURES, axis=1)
    df_model_input.to_sql(name='model_input', con=connection, if_exists='replace', index=False)
    connection.close()
    
##############################################################################
# End of utility functions module
##############################################################################
