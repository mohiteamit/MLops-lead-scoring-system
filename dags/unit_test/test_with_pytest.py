import sqlite3
import pandas as pd
import warnings

import sys
import os

from lead_scoring_data_pipeline.utils import (
    build_dbs,
    load_data_into_db,
    map_city_tier,
    map_categorical_vars,
    interactions_mapping
)

from lead_scoring_data_pipeline.constants import (
    DB_FULL_PATH
)

sys.path.insert(0, os.path.abspath("/home/airflow/dags"))

from unit_test.constants import (
    UNIT_TEST_DB_PATH,
    UNIT_TEST_DB_FILE_NAME,
    TEST_DATA_CSV_PATH
)

warnings.filterwarnings("ignore")

def test_load_data_into_db():
    load_data_into_db(TEST_DATA_CSV_PATH)
    conn = sqlite3.connect(DB_FULL_PATH)

    loaded_data = pd.read_sql('SELECT * FROM loaded_data', conn)

    conn_ut = sqlite3.connect(UNIT_TEST_DB_PATH + UNIT_TEST_DB_FILE_NAME)
    test_case = pd.read_sql('SELECT * FROM loaded_data_test_case', conn_ut)
    conn.close()
    conn_ut.close()

    assert test_case.equals(loaded_data), "Data mismatch in load_data_into_db()"

def test_map_city_tier():
    map_city_tier()

    conn = sqlite3.connect(DB_FULL_PATH)
    output_df = pd.read_sql('SELECT * FROM city_tier_mapped', conn)

    conn_ut = sqlite3.connect(UNIT_TEST_DB_PATH + UNIT_TEST_DB_FILE_NAME)
    expected_df = pd.read_sql('SELECT * FROM city_tier_mapped_test_case', conn_ut)

    conn.close()
    conn_ut.close()

    assert expected_df.equals(output_df), "Data mismatch in map_city_tier()"

def test_map_categorical_vars():
    map_categorical_vars()

    conn = sqlite3.connect(DB_FULL_PATH)
    output_df = pd.read_sql('SELECT * FROM categorical_variables_mapped', conn)

    conn_ut = sqlite3.connect(UNIT_TEST_DB_PATH + UNIT_TEST_DB_FILE_NAME)
    expected_df = pd.read_sql('SELECT * FROM categorical_variables_mapped_test_case', conn_ut)

    conn.close()
    conn_ut.close()

    assert expected_df.equals(output_df), "Data mismatch in map_categorical_vars()"

def test_interactions_mapping():
    interactions_mapping("final_output")

    conn = sqlite3.connect(DB_FULL_PATH)
    output_df = pd.read_sql('SELECT * FROM interactions_mapped', conn)

    conn_ut = sqlite3.connect(UNIT_TEST_DB_PATH + UNIT_TEST_DB_FILE_NAME)
    expected_df = pd.read_sql('SELECT * FROM interactions_mapped_test_case', conn_ut)

    conn.close()
    conn_ut.close()

    assert expected_df.equals(output_df), "Data mismatch in interactions_mapping()"
