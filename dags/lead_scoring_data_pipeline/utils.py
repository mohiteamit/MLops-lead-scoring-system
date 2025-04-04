import pandas as pd
import os
import sqlite3
from sqlite3 import Error

from lead_scoring_data_pipeline.constants import (
    DB_FULL_PATH, 
    INTERACTION_MAPPING, 
    INDEX_COLUMNS_TRAINING, 
    INDEX_COLUMNS_INFERENCE, 
    NOT_FEATURES
)

from lead_scoring_data_pipeline.mappings.significant_categorical_level import (
    PLATFORM_LEVELS, 
    MEDIUM_LEVELS, 
    SOURCE_LEVELS
)

from lead_scoring_data_pipeline.mappings.city_tier_mapping import city_tier_mapping

def build_dbs():
    '''
    Checks if the DB file exists; if not, creates one.
    Raises:
        RuntimeError: If connection to create the database fails.
    Returns:
        str: Status message.
    '''
    if os.path.isfile(DB_FULL_PATH):
        return "DB Exists"
    else:
        print("Creating Database")
        connection = None
        try:
            connection = sqlite3.connect(DB_FULL_PATH)
            print("New DB Created")
        except Error as e:
            raise RuntimeError("Failed to create the database: " + str(e))
        finally:
            if connection:
                connection.close()
        return "DB Created"

def load_data_into_db(full_path_to_csv : str):
    '''
    Loads CSV data into the DB, replacing null values in certain columns.
    Raises:
        RuntimeError: If reading CSV, connecting to DB, or writing to DB fails.
    '''
    try:
        connection = sqlite3.connect(DB_FULL_PATH)
    except Error as e:
        raise RuntimeError("Failed to connect to DB in load_data_into_db: " + str(e))
        
    try:
        df_lead_scoring = pd.read_csv(full_path_to_csv)
    except Exception as e:
        connection.close()
        raise RuntimeError("Failed to read CSV file: " + str(e))
        
    try:
        df_lead_scoring['total_leads_droppped'] = df_lead_scoring['total_leads_droppped'].fillna(0)
        df_lead_scoring['referred_lead'] = df_lead_scoring['referred_lead'].fillna(0)
        df_lead_scoring.to_sql(name='loaded_data', con=connection, if_exists='replace', index=False)
    except Exception as e:
        raise RuntimeError("Failed to load data into database: " + str(e))
    finally:
        connection.close()

def map_city_tier():
    '''
    Maps cities to their tiers, defaulting unmapped cities to 3.0.
    Raises:
        RuntimeError: If reading from or writing to DB fails.
    '''
    try:
        connection = sqlite3.connect(DB_FULL_PATH)
    except Error as e:
        raise RuntimeError("Failed to connect to DB in map_city_tier: " + str(e))
    
    try:
        df_lead_scoring = pd.read_sql('select * from loaded_data', connection)
        df_lead_scoring["city_tier"] = df_lead_scoring["city_mapped"].map(city_tier_mapping)
        df_lead_scoring["city_tier"] = df_lead_scoring["city_tier"].fillna(3.0)
        df_lead_scoring = df_lead_scoring.drop(['city_mapped'], axis=1)
        df_lead_scoring.to_sql(name='city_tier_mapped', con=connection, if_exists='replace', index=False)
    except Exception as e:
        raise RuntimeError("Failed to map city tier: " + str(e))
    finally:
        connection.close()

import os
import sqlite3
from sqlite3 import Error
import pandas as pd

def map_categorical_vars():
    '''
    Maps insignificant categorical variables to "others".
    Raises:
        RuntimeError: If reading from or writing to DB fails.
    '''
    try:
        connection = sqlite3.connect(DB_FULL_PATH)
    except Error as e:
        raise RuntimeError("Failed to connect to DB in map_categorical_vars: " + str(e))

    try:
        df_lead_scoring = pd.read_sql('select * from city_tier_mapped', connection)

        # Map insignificant platform values
        new_df = df_lead_scoring[~df_lead_scoring['first_platform_c'].isin(PLATFORM_LEVELS)].copy()
        new_df.loc[:, 'first_platform_c'] = "others"
        old_df = df_lead_scoring[df_lead_scoring['first_platform_c'].isin(PLATFORM_LEVELS)]
        df = pd.concat([new_df, old_df])

        # Map insignificant medium values
        new_df = df[~df['first_utm_medium_c'].isin(MEDIUM_LEVELS)].copy()
        new_df.loc[:, 'first_utm_medium_c'] = "others"
        old_df = df[df['first_utm_medium_c'].isin(MEDIUM_LEVELS)]
        df = pd.concat([new_df, old_df])

        # Map insignificant source values
        new_df = df[~df['first_utm_source_c'].isin(SOURCE_LEVELS)].copy()
        new_df.loc[:, 'first_utm_source_c'] = "others"
        old_df = df[df['first_utm_source_c'].isin(SOURCE_LEVELS)]
        df = pd.concat([new_df, old_df])

        # Drop duplicates and save to DB
        df = df.drop_duplicates()
        df.to_sql(name='categorical_variables_mapped', con=connection, if_exists='replace', index=False)

    except Exception as e:
        raise RuntimeError("Failed to map categorical variables: " + str(e))
    finally:
        connection.close()

def interactions_mapping(table_name : str):
    '''
    Maps interaction columns into unique interactions per provided mappings.
    Raises:
        RuntimeError: If reading from or writing to DB fails.
    '''
    try:
        connection = sqlite3.connect(DB_FULL_PATH)
    except Error as e:
        raise RuntimeError("Failed to connect to DB in interactions_mapping: " + str(e))
    
    try:
        df = pd.read_sql('select * from categorical_variables_mapped', connection)
    
        if 'app_complete_flag' in df.columns:
            index_variable = INDEX_COLUMNS_TRAINING
        else:
            index_variable = INDEX_COLUMNS_INFERENCE
        
        df_event_mapping = pd.read_csv(INTERACTION_MAPPING, index_col=[0])
        df_unpivot = pd.melt(df, id_vars=index_variable, var_name='interaction_type', value_name='interaction_value')
        df_unpivot['interaction_value'] = df_unpivot['interaction_value'].fillna(0)
        df = pd.merge(df_unpivot, df_event_mapping, on='interaction_type', how='left')
        df = df.drop(['interaction_type'], axis=1)
        df_pivot = df.pivot_table(values='interaction_value', index=index_variable, columns='interaction_mapping', aggfunc='sum')
        df_pivot = df_pivot.reset_index()
        
        df_pivot.to_sql(name='interactions_mapped', con=connection, if_exists='replace', index=False)
        
        df_model_input = df_pivot.drop(NOT_FEATURES, axis=1)
        df_model_input.to_sql(name=f'{table_name}', con=connection, if_exists='replace', index=False)

        cursor = connection.cursor()
        tables_to_drop = [
            'loaded_data', 
            'city_tier_mapped', 
            'categorical_variables_mapped', 
            'interactions_mapped'
        ]
        for table in tables_to_drop:
            cursor.execute(f"DROP TABLE IF EXISTS {table};")
            print(f"Dropped table: {table}")

    except Exception as e:
        raise RuntimeError("Failed to map interactions: " + str(e))
    finally:
        connection.close()
