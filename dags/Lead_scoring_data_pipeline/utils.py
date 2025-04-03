##############################################################################
# Import necessary modules and files
##############################################################################

import pandas as pd
import os
import sqlite3
from sqlite3 import Error

from Lead_scoring_data_pipeline.constants import CSV_FILE_NAME, DB_FILE_NAME, DB_PATH, DATA_DIRECTORY, INTERACTION_MAPPING, INDEX_COLUMNS_TRAINING, INDEX_COLUMNS_INFERENCE, NOT_FEATURES
from Lead_scoring_data_pipeline.mappings.significant_categorical_level import PLATFORM_LEVELS, MEDIUM_LEVELS, SOURCE_LEVELS
from Lead_scoring_data_pipeline.mappings.city_tier_mapping import city_tier_mapping

###############################################################################
# Define the function to build database
###############################################################################

def build_dbs():
    '''
    This function checks if the db file with specified name is present 
    in the /Assignment/01_data_pipeline/scripts folder. If it is not present it creates 
    the db file with the given name at the given path. 
    '''
    db_full_path = os.path.join(DB_PATH, DB_FILE_NAME)
    if os.path.isfile(db_full_path):
        print("DB Already Exsist")
        return "DB Exsists"
    else:
        print("Creating Database")
        #create a database connection to a SQLite database
        connection = None
        try:
            connection = sqlite3.connect(db_full_path)
            print("New DB Created")
        except Error as e:
            print(e)
            return "Error"
        finally:
            if connection:
                connection.close()
                return "DB Created"

###############################################################################
# Define function to load the csv file to the database
##############################################################################

def load_data_into_db():
    '''
    Thie function loads the data present in data directory into the db
    which was created previously.
    It also replaces any null values present in 'toal_leads_dropped' and
    'referred_lead' columns with 0.
    '''
    db_full_path = os.path.join(DB_PATH, DB_FILE_NAME)
    connection = sqlite3.connect(db_full_path)
    
    csv_full_path = os.path.join(DATA_DIRECTORY, CSV_FILE_NAME)
    df_lead_scoring = pd.read_csv(csv_full_path)

    df_lead_scoring['total_leads_droppped'] = df_lead_scoring['total_leads_droppped'].fillna(0)
    df_lead_scoring['referred_lead'] = df_lead_scoring['referred_lead'].fillna(0)

    df_lead_scoring.to_sql(name='loaded_data', con=connection, if_exists='replace', index=False)

    connection.close()


###############################################################################
# Define function to map cities to their respective tiers
##############################################################################
    
def map_city_tier():
    '''
    This function maps all the cities to their respective tier as per the
    mappings provided in the city_tier_mapping.py file. If a
    particular city's tier isn't mapped(present) in the city_tier_mapping.py 
    file then the function maps that particular city to 3.0 which represents
    tier-3.
    '''
    db_full_path = os.path.join(DB_PATH, DB_FILE_NAME)
    connection = sqlite3.connect(db_full_path)
    
    df_lead_scoring = pd.read_sql('select * from loaded_data', connection)

    df_lead_scoring["city_tier"] = df_lead_scoring["city_mapped"].map(city_tier_mapping)
    df_lead_scoring["city_tier"] = df_lead_scoring["city_tier"].fillna(3.0)
    df_lead_scoring = df_lead_scoring.drop(['city_mapped'], axis=1)

    df_lead_scoring.to_sql(name='city_tier_mapped', con=connection, if_exists='replace', index=False)

    connection.close()

###############################################################################
# Define function to map insignificant categorial variables to "others"
###############################################################################

def map_categorical_vars():
    '''
    This function maps all the insignificant variables present in 'first_platform_c'
    'first_utm_medium_c' and 'first_utm_source_c'. The list of significant variables
    should be stored in a python file in the 'significant_categorical_level.py' 
    so that it can be imported as a variable in utils file.
    '''
    db_full_path = os.path.join(DB_PATH, DB_FILE_NAME)
    connection = sqlite3.connect(db_full_path)

    df_lead_scoring = pd.read_sql('select * from city_tier_mapped', connection)

    new_df = df_lead_scoring[~df_lead_scoring['first_platform_c'].isin(PLATFORM_LEVELS)]
    new_df['first_platform_c'] = "others"
    old_df = df_lead_scoring[df_lead_scoring['first_platform_c'].isin(PLATFORM_LEVELS)]  
    df = pd.concat([new_df, old_df])

    new_df = df[~df['first_utm_medium_c'].isin(MEDIUM_LEVELS)]
    new_df['first_utm_medium_c'] = "others"
    old_df = df[df['first_utm_medium_c'].isin(MEDIUM_LEVELS)]
    df = pd.concat([new_df, old_df])

    new_df = df[~df['first_utm_source_c'].isin(SOURCE_LEVELS)]
    new_df['first_utm_source_c'] = "others"
    old_df = df[df['first_utm_source_c'].isin(SOURCE_LEVELS)]
    df = pd.concat([new_df, old_df])

    df = df.drop_duplicates()

    df.to_sql(name='categorical_variables_mapped', con=connection, if_exists='replace', index=False)
    connection.close()


###############################################################################
# Define function that maps interaction columns into 4 types of interactions
###############################################################################
def interactions_mapping():
    '''
    This function maps the interaction columns into 4 unique interaction columns
    These mappings are present in 'interaction_mapping.csv' file. 
    '''
    db_full_path = os.path.join(DB_PATH, DB_FILE_NAME)
    connection = sqlite3.connect(db_full_path)

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
    df_model_input.to_sql(name='model_input', con=connection, if_exists='replace', index=False)
    
    connection.close()
