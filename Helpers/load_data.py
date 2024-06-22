"""
This module contains the functions for loading and processing JSON dataset files.

Functions:
    load_json(path_to_dataset: str) -> dict:
        Loads a JSON file and returns it as a dict.

    load_data_from_json(parsed_json: dict) -> pd.DataFrame:
        Builds a pd.DataFrame from the parsed smaller dataset and returns it.
    
    load_data_from_larger_json(parsed_json: dict) -> pd.DataFrame:
        Builds a pd.DataFrame from the parsed larger dataset and returns it.
"""

import json
import os

import pandas as pd


def load_json(path_to_dataset: str) -> dict:
    """
        Loads a JSON file and returns it as a dict.

    Args:
        path_to_dataset (str): The file path to the JSON dataset.

    Returns:
        pd.DataFrame: The DataFrame containing the processed data.
    """

    if os.access(path_to_dataset, os.R_OK):
        print("File is readable")
        with open(path_to_dataset) as file:
            parsed_json = json.load(file)
    else:
        print("File is not readable")
    return parsed_json


def load_data_from_smaller_json(parsed_json: dict) -> pd.DataFrame:
    """
    Builds a pd.DataFrame from the parsed smaller dataset and returns it.

    Args:
        parsed_json (dict): A dict containing the previously parsed dataset.

    Returns:
        pd.DataFrame: A DataFrame containing the processed data.
    """

    all_records = pd.DataFrame()
    json_subarray_name = 'traces'
    for i in pd.json_normalize(parsed_json[json_subarray_name]):
        single_record = pd.json_normalize(parsed_json[json_subarray_name][i])
        all_records = pd.concat(
            [all_records, single_record], ignore_index=True)
    return all_records


def load_data_from_larger_json(parsed_json: dict) -> pd.DataFrame:
    """
    Builds a pd.DataFrame from the parsed larger dataset and returns it.

    Args:
        parsed_json (dict): A dict containing the previously parsed dataset.

    Returns:
        pd.DataFrame: A DataFrame containing the processed data.
    """

    all_records = pd.DataFrame()
    json_subarray_name = 'traces'
    for i in pd.json_normalize(parsed_json[json_subarray_name]):
        single_record = pd.json_normalize(parsed_json[json_subarray_name][i])
        single_record_filtered = single_record.dropna(how='all', axis=1)
        all_records = pd.concat(
            [all_records, single_record_filtered], ignore_index=True)

    return all_records
