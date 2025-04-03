###############################################################################
# Import necessary modules
###############################################################################

import mlflow
import pandas as pd
import sqlite3
import os
from datetime import datetime

from lead_scoring_inference_pipeline.constants import DB_FULL_PATH, FEATURES_TO_ENCODE, ONE_HOT_ENCODED_FEATURES, MLFLOW_TRACKING_URI, MODEL_NAME, STAGE, PREDICTION_DIST_TXT

###############################################################################
# Define the function to train the model
###############################################################################

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
    input_data_df = pd.read_sql('SELECT * FROM MODEL_INPUT', conn)

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
            msg = f"Feature '{feature}' not found in MODEL_INPUT dataframe."
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
    # if 'app_complete_flag' not in encoded_features_df.columns:
    #     msg = "Target feature 'app_complete_flag' not found in dataframe."
    #     print(msg)
    #     conn.close()
    #     raise ValueError(msg)

    features_to_save_df = encoded_features_df.drop(['app_complete_flag'], axis=1)
    # target_to_save_df = encoded_features_df[['app_complete_flag']]
    features_to_save_df.to_sql(name='INFERENCE', con=conn, if_exists='replace', index=False)
    # target_to_save_df.to_sql(name='TARGET', con=conn, if_exists='replace', index=False)

    conn.close()

###############################################################################
# Define the function to load the model from mlflow model registry
###############################################################################

def get_models_prediction():
    '''
    This function loads the model which is in production from mlflow registry and 
    uses it to do prediction on the input dataset. Please note this function will load
    the latest version of the model present in the production stage. 

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be
        model from mlflow model registry
        model name: name of the model to be loaded
        stage: stage from which the model needs to be loaded i.e. production

    OUTPUT
        Store the predicted values along with input data into a table

    SAMPLE USAGE
        load_model()
    '''
    # set the tracking uri
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # load the latest model from production stage
    loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{STAGE}")

    # read the new data
    conn = sqlite3.connect(os.path.join(DB_FULL_PATH))
    df_fea_inf = pd.read_sql('SELECT * FROM INFERENCE', conn)

    # run the model to generate the prediction on new data
    y_pred = loaded_model.predict(df_fea_inf)
    df_fea_inf['pred_app_complete_flag'] = y_pred

    # store the data in a table
    df_fea_inf.to_sql(name='PREDICTIONS', con=conn, if_exists='replace', index=False)
    conn.close()

###############################################################################
# Define the function to check the distribution of output column
###############################################################################

def prediction_ratio_check():
    '''
    This function calculates the % of 1 and 0 predicted by the model and  
    writes it to a file named 'prediction_distribution.txt'. This file 
    should be created in the ~/airflow/dags/Lead_scoring_inference_pipeline 
    folder. 
    This helps us to monitor if there is any drift observed in the predictions 
    from our model at an overall level. This would determine our decision on 
    when to retrain our model.
    
    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be

    OUTPUT
        Write the output of the monitoring check in prediction_distribution.txt with 
        timestamp.

    SAMPLE USAGE
        prediction_ratio_check()
    '''
    # read the input data
    conn = sqlite3.connect(os.path.join(DB_FULL_PATH))
    df_pred_val = pd.read_sql('SELECT * FROM PREDICTIONS', conn)

    # get the distribution of categories in prediction col
    val_cnts = df_pred_val['pred_app_complete_flag'].value_counts(normalize=True)

    # write the output in a file
    ct = datetime.now()
    st = str(ct) + ' %of 1 = ' + str(val_cnts[1]) + ' %of 2 = ' + str(val_cnts[0])
    with open(os.path.join(PREDICTION_DIST_TXT), 'a') as f:
        f.write(st + "\n")
    conn.close()

###############################################################################
# Define the function to check the columns of input features
###############################################################################

def input_features_check():
    '''
    This function checks whether all the input columns are present in our new
    data. This ensures the prediction pipeline doesn't break because of change in
    columns in input data.

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be
        ONE_HOT_ENCODED_FEATURES: List of all the features which need to be present
        in our input data.

    OUTPUT
        It writes the output in a log file based on whether all the columns are present
        or not.
        1. If all the input columns are present then it logs - 'All the models input are present'
        2. Else it logs 'Some of the models inputs are missing'

    SAMPLE USAGE
        input_features_check()
    '''
    # read the input data
    conn = sqlite3.connect(os.path.join(DB_FULL_PATH))
    df_inf = pd.read_sql('SELECT * FROM INFERENCE', conn)

    # check if all columns are present
    check = set(df_inf.columns) == set(ONE_HOT_ENCODED_FEATURES)

    if check:
        print('All the models input are present')
    else:
        print('Some of the models inputs are missing')
    conn.close()
