DB_FILE_NAME = 'utils_output.db'
UNIT_TEST_DB_FILE_NAME = 'unit_test_cases.db'
DB_PATH = "/home/Assignment/01_data_pipeline/scripts/unit_test/"
DATA_DIRECTORY = '/home/Assignment/01_data_pipeline/scripts/unit_test/'

INTERACTION_MAPPING = "interaction_mapping.csv"
INDEX_COLUMNS_TRAINING = ['created_date', 'city_tier', 'first_platform_c',
                'first_utm_medium_c', 'first_utm_source_c', 'total_leads_droppped',
                'referred_lead', 'app_complete_flag']

INDEX_COLUMNS_INFERENCE = ['created_date', 'city_tier', 'first_platform_c',
                'first_utm_medium_c', 'first_utm_source_c', 'total_leads_droppped',
                'referred_lead']

NOT_FEATURES = ['created_date', 'assistance_interaction', 'career_interaction',
                'payment_interaction', 'social_interaction', 'syllabus_interaction']

import pandas as pd
import os
import sqlite3
from sqlite3 import Error

from constants import *
from significant_categorical_level import *
from city_tier_mapping import city_tier_mapping

def build_dbs():
    if os.path.isfile(DB_PATH+DB_FILE_NAME):
        print("DB Already Exsist")
        return "DB Exsists"
    else:
        print("Creating Database")
        #create a database connection to a SQLite database
        connection = None
        try:
            connection = sqlite3.connect(DB_PATH+DB_FILE_NAME)
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
# ##############################################################################

def load_data_into_db():
    connection = sqlite3.connect(DB_PATH+DB_FILE_NAME)
    
    df_lead_scoring = pd.read_csv(DATA_DIRECTORY+'leadscoring_test.csv')

    df_lead_scoring['total_leads_droppped'] = df_lead_scoring['total_leads_droppped'].fillna(0)
    df_lead_scoring['referred_lead'] = df_lead_scoring['referred_lead'].fillna(0)

    df_lead_scoring.to_sql(name='loaded_data', con=connection, if_exists='replace', index=False)

    connection.close()


###############################################################################
# Define function to map cities to their respective tiers
# ##############################################################################

    
def map_city_tier():
    connection = sqlite3.connect(DB_PATH+DB_FILE_NAME)
    
    df_lead_scoring = pd.read_sql('select * from loaded_data', connection)

    df_lead_scoring["city_tier"] = df_lead_scoring["city_mapped"].map(city_tier_mapping)
    df_lead_scoring["city_tier"] = df_lead_scoring["city_tier"].fillna(3.0)
    df_lead_scoring = df_lead_scoring.drop(['city_mapped'], axis=1)

    df_lead_scoring.to_sql(name='city_tier_mapped', con=connection, if_exists='replace', index=False)

    connection.close()

###############################################################################
# Define function to map insignificant categorial variables to "others"
# ##############################################################################


def map_categorical_vars():
    connection = sqlite3.connect(DB_PATH+DB_FILE_NAME)

    df_lead_scoring = pd.read_sql('select * from city_tier_mapped', connection)

    new_df = df_lead_scoring[~df_lead_scoring['first_platform_c'].isin(list_platform)]
    # replace the value of these levels to others
    new_df['first_platform_c'] = "others"
    # get rows for levels which are present in list_platform
    old_df = df_lead_scoring[df_lead_scoring['first_platform_c'].isin(list_platform)]  
    # concatenate new_df and old_df to get the final dataframe
    df = pd.concat([new_df, old_df])

    # get rows for levels which are not present in list_medium
    new_df = df[~df['first_utm_medium_c'].isin(list_medium)]
    # replace the value of these levels to others
    new_df['first_utm_medium_c'] = "others"
    # get rows for levels which are present in list_medium
    old_df = df[df['first_utm_medium_c'].isin(list_medium)]
    # concatenate new_df and old_df to get the final dataframe
    df = pd.concat([new_df, old_df])

    # get rows for levels which are not present in list_source
    new_df = df[~df['first_utm_source_c'].isin(list_source)]
    # replace the value of these levels to others
    new_df['first_utm_source_c'] = "others"
    # get rows for levels which are present in list_source
    old_df = df[df['first_utm_source_c'].isin(list_source)]
    # concatenate new_df and old_df to get the final dataframe
    df = pd.concat([new_df, old_df])

    df = df.drop_duplicates()

    df.to_sql(name='categorical_variables_mapped', con=connection, if_exists='replace', index=False)
    connection.close()


##############################################################################
# Define function that maps interaction columns into 4 types of interactions
# #############################################################################
def interactions_mapping():
    connection = sqlite3.connect(DB_PATH+DB_FILE_NAME)

    df = pd.read_sql('select * from categorical_variables_mapped', connection)
    
    if 'app_complete_flag' in df.columns:
        index_variable = INDEX_COLUMNS_TRAINING
    else:
        index_variable = INDEX_COLUMNS_INFERENCE
        
    df_event_mapping = pd.read_csv(INTERACTION_MAPPING, index_col=[0])
    df_unpivot = pd.melt(df, id_vars=index_variable, var_name='interaction_type', value_name='interaction_value')
    df_unpivot['interaction_value'] = df_unpivot['interaction_value'].fillna(0)
    df = pd.merge(df_unpivot, df_event_mapping,on='interaction_type', how='left')
    df = df.drop(['interaction_type'], axis=1)
    df_pivot = df.pivot_table(values='interaction_value', index=index_variable, columns='interaction_mapping', aggfunc='sum')
    df_pivot = df_pivot.reset_index()
    
    df_pivot.to_sql(name='interactions_mapped', con=connection, if_exists='replace', index=False)
    
    df_model_input = df_pivot.drop(NOT_FEATURES, axis=1)
    df_model_input.to_sql(name='model_input', con=connection, if_exists='replace', index=False)
    
    connection.close()
   