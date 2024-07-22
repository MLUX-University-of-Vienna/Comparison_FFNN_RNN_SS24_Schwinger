"""
This module contains the functions to preprocess the pd.DataFrame before splitting it into feature vectors.

Functions:
    remove_columns_with_a_single_value(original_dataframe: pd.DataFrame) -> pd.DataFrame:
        Removes all columns that only contain one possible value and therefore do not provide any useful information for the neural network.
        Columns with a single unique do not provide any information for the neural network, since every possible sample will have the same value.
    
    remove_low_appearance_values(original_dataframe: pd.DataFrame, count_threshold: int, column_name: str) -> pd.DataFrame:
        Calculates how often each unique values appears in a specific column of the {original_dataframe}.
        Rows in the {original_dataframe} with unique values appearing below the {count_threshold} will get removed from the DataFrame.
    
    number_of_unique_values_per_column(dataframe: pd.DataFrame) -> list[(str, int)]:
        Returns a list of tuples containing column names and the amount of unique values in this column.

    replace_null_values_in_one_column_with_a_placeholder(original_dataframe: pd.DataFrame, dataframe_column_name: str, parsed_json: dict, json_subarray_name: str):
        Replaces null (nan) values in a specific column with a placeholder.
        Assumes the null value is a float nan and not a string 'null'
        Only works if the column in the dataframe has an enum defining how to transform the value into an integer value.
        For example: event_type. The event_type has a trace_enum entry listing all possible different values.
        This method gets this list and calculates the individual number of possible entries. 
        The placeholder is the highest int from the encoding dict + 1.
        The placeholder is added as integer value.

    replace_null_values_with_column_mode(original_dataframe: pd.DataFrame) -> pd.DataFrame:
        Returns the DataFrame without any null value. 
        Each null value gets substituted with the mode value of the column.
        The mode is the value that appears with the highest frequency in the column.
        The null values get substituted, because a neural network needs non-null integer values as input features. 

    move_column_to_the_front(original_dataframe: pd.DataFrame, column_to_move: str) -> pd.DataFrame:
        Returns the DataFrame with the {column_to_move} moved at the first index of the DataFrame columns.
    
    add_prev_and_prev_prev_event(original_dataframe: pd.DataFrame, unique_classifications: int, event_column_name: str, session_column_name: str) -> pd.DataFrame:
        Returns the DataFrame with the prev_event and the prev_prev_event feature.

    get_weather_data(original_dataframe: pd.DataFrame, column_to_encode: str, json_subarray_name: str, parsed_json: dict) -> pd.DataFrame:
        Returns the DataFrame with more detailed weather data. 

    define_columns_to_drop(original_dataframe: pd.DataFrame) -> list[str]:
        Returns a list of all absolute columns from a predefined list that are presented in the DataFrame.

    define_ohe_features_list(original_dataframe: pd.DataFrame) -> list[str]:
        Returns a list of all one hot encoded features from a predefined list that are presented in the DataFrame.

    define_ordinal_features_list(original_dataframe: pd.DataFrame) -> list[str]:
        Returns a list of all ordinal features from a predefined list that are presented in the DataFrame.

    encode_cyclic_feature(original_dataframe: pd.DataFrame, column_to_encode: str, period: int) -> pd.DataFrame:    
        Encodes a cyclical feature in a DataFrame column using a cosine and sine transformation, 
        to let the neural network correctly learn periodic patterns.

    encode_utc_timestamps_with_two_different_patterns(original_dataframe: pd.DataFrame, column_to_encode: str, patterns: list[str]) -> pd.DataFrame:
        Encodes a utc timestamp in a DataFrame column to a unix timestamp, to convert the timestamp into an integer, 
        that can be used as input feature for the neural network.

        Iterates over all entries in the {column_to_encode}, transforms them into an unix timestamp.
        These unix timestamps then get added to a list containing all timestamps. 
        After iterating over all entries the values in the {column_to_encode} get substituted with the list of timestamps. 

        Used for columns with two different possible utc patterns.

    encode_utc_timestamp(original_dataframe: pd.DataFrame, column_to_encode: str, pattern: str) -> pd.DataFrame:
        Encodes a utc timestamp in a DataFrame column to a unix timestamp, to convert the timestamp into an integer, 
        that can be used as input feature for the neural network.

        Iterates over all entries in the {column_to_encode}, transforms them into an unix timestamp.
        These unix timestamps then get added to a list containing all timestamps. 
        After iterating over all entries the values in the {column_to_encode} get substituted with the list of timestamps. 

    encode_boolean(original_dataframe: pd.DataFrame, column_to_encode: str) -> pd.DataFrame:
        Transforms the content of column with boolean string representation into a binary integer representation.

    encode_date(original_dataframe: pd.DataFrame, column_to_encode: str, pattern: str) -> pd.DataFrame:
        Transforms a Date as string with the specified string pattern into a ordinal encoded representation.
        Ordinal encoded integers imply an ordering. 

    encode_weather_enums(original_dataframe: pd.DataFrame, json_subarray_name: str, json_key: str, parsed_json: dict) -> pd.DataFrame:
        Transforms the string from the weather_hour data into the in the JSON defined integer encoded representation of the enum.
        The same representation is also used in the weather_day data. 

    encode_weather_data(original_dataframe: pd.DataFrame, weather_id_column: str) -> pd.DataFrame:
        Returns the DataFrame with encoded weather data. 

    onehot_encode_column(original_dataframe: pd.DataFrame, column_to_encode: str) -> pd.DataFrame:
        One-hot encodes the passed column of the DataFrame. 

        One-hot encoding is a technique used to convert categorical variables into a numerical representation,
        which is necessary to use nominal non-integer features without an inherent order as input in a neural network.
        This guarantees that the neural network can correctly learn from the data without implying any ordinal relationships.

        One-hot encoding splits the specified column by creating new binary columns for each unique value present in the column.
        Each binary column represents if the corresponding value is present for the row in the DataFrame.

    label_encode_column(original_dataframe: pd.DataFrame, column_to_encode: str) -> pd.DataFrame:
        Label encodes the passed column of the DataFrame.

        Label encoding is a technique used to convert categorical variables into a numerical representation,
        which is necessary to use non-integer features as input in a neural network.

        Label encoding does transform the categorical variable into a ordinal representation and therefore implies an order.         

    binary_encode_column(original_dataframe: pd.DataFrame, column_to_encode: str) -> pd.DataFrame: 
        Binary encodes the passed column of the DataFrame.

        Binary encoding is a technique used to convert categorical variables into a binary representation,
        which is necessary to use non-integer features as input in a neural network.

        Binary encoding first transforms the categorical variables into a numerical representation.
        Afterwards this numerical representation is transformed into its binary number and split into multiple columns. 

        Binary encoding can imply a weak ordering.
"""


from datetime import datetime

import category_encoders
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def remove_columns_with_a_single_value(original_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Removes all columns that only contain one possible value and therefore do not provide any useful information for the neural network.
    Columns with a single unique do not provide any information for the neural network, since every possible sample will have the same value. 

    Args:
        original_dataframe (pd.DataFrame): The input DataFrame to check for columns with single unique values.

    Returns: 
        pd.DataFrame: A modified version of the input DataFrame.
        Note: All columns with a single unique value are dropped.
    """
    return original_dataframe[[column for column in original_dataframe.columns if len(original_dataframe[column].unique()) > 1]]


def remove_low_appearance_values(original_dataframe: pd.DataFrame, count_threshold: int, column_name: str) -> pd.DataFrame:
    """
    Calculates how often each unique values appears in a specific column of the {original_dataframe}.
    Rows in the {original_dataframe} with unique values appearing below the {count_threshold} will get removed from the DataFrame. 

    Args:   
        original_dataframe (pd.DataFrame): The input dataframe to be modified.
        count_threshold (int): All unique values with a value_count below this number will be removed.
        column_name (str): The name of the column in the DataFrame on which the value_count for each unique value is calculated.

    Returns:
        pd.DataFrame: A modified version of the input DataFrame with all classifications below the defined threshold removed.
    """
    appearance_count = original_dataframe[column_name].value_counts()
    return original_dataframe[~original_dataframe[column_name].isin(appearance_count[appearance_count < count_threshold].index.tolist())]


def number_of_unique_values_per_column(dataframe: pd.DataFrame) -> list[(str, int)]:
    """ 
    Returns a list of tuples containing column names and the amount of unique values in this column.

    Args:  
        original_dataframe (pd.DataFrame): The input DataFrame to determine the number of unique values per column.

    Returns:
        list[(str, int)]: A list of tuples where each tuple contains:
            - The column name as a string.
            - The amount of unique values in that column as an integer.
    """
    return [(column, dataframe[column].nunique()) for column in dataframe.columns]


def replace_null_values_in_one_column_with_a_placeholder(original_dataframe: pd.DataFrame, dataframe_column_name: str, parsed_json: dict, json_subarray_name: str):
    """
    Replaces null (nan) values in a specific column with a placeholder.
    Assumes the null value is a float nan and not a string 'null'
    Only works if the column in the dataframe has an enum defining how to transform the value into an integer value.
    For example: event_type. The event_type has a trace_enum entry listing all possible different values.
    This method gets this list and calculates the individual number of possible entries. 
    The placeholder is the highest int from the encoding dict + 1.
    The placeholder is added as integer value.

    Args:
        original_dataframe (pd.DataFrame):
        dataframe_column_name (str):
        parsed_json (dict):
        json_subarray_name (str):

    Returns:
        pd.DataFrame: A modified version of the input DataFrame where all null values are substituted with a placeholder.
    """

    unique_types = parsed_json[json_subarray_name][dataframe_column_name]

    for i in range(0, len(original_dataframe)):
        if pd.isna(original_dataframe.iloc[i][dataframe_column_name]):
            original_dataframe.at[original_dataframe.index[i],
                                  dataframe_column_name] = len(unique_types)

    original_dataframe[dataframe_column_name] = original_dataframe[dataframe_column_name].astype(
        int)
    return original_dataframe


def replace_null_values_with_column_mode(original_dataframe: pd.DataFrame) -> pd.DataFrame:
    """ 
    Returns the DataFrame without any null value. 
    Each null value gets substituted with the mode value of the column.
    The mode is the value that appears with the highest frequency in the column.
    The null values get substituted, because a neural network needs non-null integer values as input features. 

    Args:
        original_dataframe (pd.DataFrame): The input DataFrame to modify.

    Returns:
        pd.DataFrame: A modified version of the input DataFrame without null values.
    """
    original_dataframe = original_dataframe.infer_objects(copy=False)
    null_columns = original_dataframe.columns[original_dataframe.isnull(
    ).any()]

    for column in null_columns:
        original_dataframe[column] = original_dataframe[column].fillna(
            original_dataframe[column].mode()[0])
    return original_dataframe


def move_column_to_the_front(original_dataframe: pd.DataFrame, column_to_move: str) -> pd.DataFrame:
    """ 
    Returns the DataFrame with the {column_to_move} moved at the first index of the DataFrame columns.

    Args:
        original_dataframe (pd.DataFrame): The input DataFrame to modify.
        column_to_move (str): The name of the column to move to the beginning of the input DataFrame.

    Returns:
        pd.DataFrame: A modified version of the input DataFrame with the {column_to_move} at the beginning of the DataFrame.
    """
    first_column = original_dataframe.pop(column_to_move)
    original_dataframe.insert(0, column_to_move, first_column)
    return original_dataframe


def add_prev_and_prev_prev_event(original_dataframe: pd.DataFrame, unique_classifications: int, event_column_name: str, session_column_name: str) -> pd.DataFrame:
    """ 
    Returns the DataFrame with the prev_event and the prev_prev_event feature.

    Args:
        original_dataframe (pd.DataFrame): The input DataFrame to modify.
        unique_classifications (int): Number of unique classifications including the placeholder for unknown labels.
        event_column_name (str): Name of the column which prev and prev_prev features should be added.
        session_column_name (str): Name of the column that saves the session.

    Returns:
        pd.DataFrame: A modified version of the input DataFrame with added prev_event and prev_prev_event features.
    """
    prev_event = []
    prev_prev_event = []

    for i in range(0, len(original_dataframe)):
        if i == 0:
            prev_event.append(unique_classifications)
            prev_prev_event.append(unique_classifications)

        elif i == 1:
            prev_event.append(original_dataframe.iloc[i-1][event_column_name])
            prev_prev_event.append(unique_classifications)

        elif original_dataframe.iloc[i-2][session_column_name] == original_dataframe.iloc[i-1][session_column_name] == original_dataframe.iloc[i][session_column_name]:
            prev_event.append(original_dataframe.iloc[i-1][event_column_name])
            prev_prev_event.append(
                original_dataframe.iloc[i-2][event_column_name])

        elif original_dataframe.iloc[i-1][session_column_name] == original_dataframe.iloc[i][session_column_name]:
            prev_event.append(original_dataframe.iloc[i-1][event_column_name])
            prev_prev_event.append(unique_classifications)

        elif original_dataframe.iloc[i-1][session_column_name] != original_dataframe.iloc[i][session_column_name]:
            prev_event.append(unique_classifications)
            prev_prev_event.append(unique_classifications)

    original_dataframe.insert(
        0, f"prev_prev_{event_column_name}", prev_prev_event, allow_duplicates=True)
    original_dataframe.insert(
        0, f"prev_{event_column_name}", prev_event, allow_duplicates=True)
    return original_dataframe


def get_weather_data(original_dataframe: pd.DataFrame, column_to_encode: str, json_subarray_name: str, parsed_json: dict) -> pd.DataFrame:
    """ 
    Returns the DataFrame with more detailed weather data. 

    Args: 
        original_dataframe (pd.DataFrame): The input DataFrame to modify.
        column_to_encode (str):  The name of the column from the input DataFrame with the ID to fetch the weather data.
        json_subarray_name (str): The name of the subarray in the json that contains the in-depth weather data for the ID included in the input DataFrame.
        parsed_json(dict):  A dict containing the parsed Dataset.  
    Returns:
        pd.DataFrame: A modified version of the input DataFrame with the weather_data included.
        Note: All columns with only a single unique value get removed and the column with the ID ({column_to_encode}) also gets dropped. 
    """
    weather_data_list = []

    for _, row in original_dataframe.iterrows():
        # get the id from the original dataframe
        weather_id = row[column_to_encode]
        # get the enum dict corresponding to the id
        weather_day_data = parsed_json[json_subarray_name][str(weather_id)]
        # add the {column to encode} to the key since weather_day and weather_hour share keys
        weather_day_data = {f"{column_to_encode} {k}": v for k,
                            v in weather_day_data.items() if k != 'id'}
        weather_data_list.append(weather_day_data)
    weather_data_df = pd.DataFrame(
        weather_data_list, index=original_dataframe.index)
    weather_data_df = remove_columns_with_a_single_value(weather_data_df)
    original_dataframe = pd.concat(
        [original_dataframe, weather_data_df], axis=1)

    # drop the id column since we already fetched the data and do not need it anymore
    original_dataframe = original_dataframe.drop(columns=column_to_encode)

    return original_dataframe


def define_columns_to_drop(original_dataframe: pd.DataFrame) -> list[str]:
    """
    Returns a list of all absolute columns from a predefined list that are presented in the DataFrame.

    Args:
        original_dataframe (pd.DataFrame): The DataFrame, which columns are compared to the predefined list of absolute values.

    Returns:
        list[str]: A list containing all the columns that are going to be dropped.
    """
    possible_columns_to_drop = ['weather_day_id moon_set', 'weather_day_id moon_rise', 'content_portal', 'device_online', 'time_utc', 'time_local',
                                'device_height_px', 'device_width_px', 'weather_day_id sun_set', 'weather_day_id sun_rise', 'weather_day_id moon_set',
                                'weather_day_id moon_rise', 'weather_day_id created_at', 'weather_day_id calculated_at', 'weather_day_id forecast_date',
                                'weather_hour_id created_at', 'weather_hour_id calculated_at', 'weather_hour_id forecast_time', 'event_data.for_date']
    columns_in_dataframe = [col for col in original_dataframe.columns]
    return [col for col in possible_columns_to_drop if col in columns_in_dataframe]


def define_ohe_features_list(original_dataframe: pd.DataFrame) -> list[str]:
    """
    Returns a list of all one hot encoded features from a predefined list that are presented in the DataFrame.

    Args:
        original_dataframe (pd.DataFrame): The DataFrame, which columns are compared to the predefined list of ordinal values.

    Returns:
        list[str]: A list containing all the columns that are going to be one hot encoded.
    """
    possible_ohe_features = ['device_class', 'device_orientation', 'oha_language_iso2', 'oha_layout',
                             'device_country_iso2', 'device_language_iso2', 'event_type', 'device_platform']

    features_in_dataframe = [col for col in original_dataframe.columns]
    return [feature for feature in possible_ohe_features if feature in features_in_dataframe]


def define_ordinal_features_list(original_dataframe: pd.DataFrame) -> list[str]:
    """
    Returns a list of all ordinal features from a predefined list that are presented in the DataFrame.

    Args:
        original_dataframe (pd.DataFrame): The DataFrame, which columns are compared to the predefined list of ordinal values.

    Returns:
        list[str]: A list containing all the columns that are going to be scaled.
    """
    possible_ordinal_features = ['time_dow_sin', 'time_dow_cos', 'time_hod_sin', 'time_hod_cos', 'weather_hour_id thunderstorm_prob', 'weather_day_id thunderstorm_prob',
                                 'weather_hour_id wind_direction_sin', 'weather_hour_id wind_direction_cos', 'weather_day_id sunshine_h', 'weather_day_id temp_max_c',
                                 'weather_day_id temp_min_c', 'weather_day_id moon_phase_sin', 'weather_day_id prec_prob_pct', 'weather_day_id prec_rain_mm_h',
                                 'weather_day_id prec_snow_mm_h', 'weather_day_id wind_speed_kmh', 'weather_day_id prec_total_mm_h', 'weather_day_id temp_felt_max_c',
                                 'weather_day_id temp_felt_min_c', 'weather_day_id humidity_mean_pct', 'weather_day_id wind_speed_max_kmh', 'weather_day_id cloud_cover_max_pct',
                                 'weather_day_id cloud_cover_min_pct', 'weather_day_id cloud_cover_mean_pct', 'weather_hour_id temp_c', 'weather_hour_id sunshine_h',
                                 'weather_hour_id temp_felt_c', 'weather_hour_id humidity_pct', 'weather_hour_id prec_rain_mm_h', 'weather_hour_id prec_snow_mm_h',
                                 'weather_day_id wind_direction_sin', 'weather_day_id wind_direction_cos', 'weather_hour_id wind_speed_kmh', 'weather_hour_id cloud_cover_pct',
                                 'weather_hour_id prec_total_mm_h', 'weather_hour_id forecast_distance_h', 'weather_hour_id wind_strength', 'weather_day_id wind_strength',
                                 'weather_day_id moon_phase_cos', 'weather_hour_id prec_snow_mm_h', 'weather_day_id wind_strength', 'weather_day_id prec_snow_mm_h',
                                 'weather_hour_id wind_strength', 'weather_hour_id forecast_distance_h']
    features_in_dataframe = [col for col in original_dataframe.columns]
    return [feature for feature in possible_ordinal_features if feature in features_in_dataframe]


def encode_cyclic_feature(original_dataframe: pd.DataFrame, column_to_encode: str, period: int) -> pd.DataFrame:
    """
    Encodes a cyclical feature in a DataFrame column using a cosine and sine transformation, 
    to let the neural network correctly learn periodic patterns.

    The original feature (e.g. time, date) gets split up into two new columns.
    A cosine and a sine column while the original column gets dropped.

    Args:
        original_dataframe (pd.DataFrame): The input DataFrame to modify the cyclical features column. 
        column_to_encode (str): The name of the column to modify in the input DataFrame.
        period (int):  The timeframe of a single cycle in the column to be encoded. 

    Returns:
        pd.DataFrame: A modified version of the input DataFrame with the {column_to_encode} replaced by two new columns:
            - {column_to_encode}_sin: The sine-transformed cyclical feature.
            - {column_to_encode}_cos: The cosine-transformed cyclical feature.
        Note: The {column_to_encode} is dropped from the DataFrame after encoding.
    """
    original_dataframe[column_to_encode] = original_dataframe[column_to_encode].astype(
        'float64')
    original_dataframe[column_to_encode + '_sin'] = np.sin(
        2 * np.pi * original_dataframe[column_to_encode] / period)
    original_dataframe[column_to_encode + '_cos'] = np.cos(
        2 * np.pi * original_dataframe[column_to_encode] / period)
    original_dataframe = original_dataframe.drop(columns=column_to_encode)
    return original_dataframe


def encode_utc_timestamps_with_two_different_patterns(original_dataframe: pd.DataFrame, column_to_encode: str, patterns: list[str]) -> pd.DataFrame:
    """ 
    Encodes a utc timestamp in a DataFrame column to a unix timestamp, to convert the timestamp into an integer, 
    that can be used as input feature for the neural network.

    Iterates over all entries in the {column_to_encode}, transforms them into an unix timestamp.
    These unix timestamps then get added to a list containing all timestamps. 
    After iterating over all entries the values in the {column_to_encode} get substituted with the list of timestamps. 

    Used for columns with two different possible utc patterns.

    Args:
        original_dataframe (pd.DataFrame): The input DataFrame to encode the utc timestamp.
        column_to_encode (str): The name of the column to modify in the input DataFrame.
        patterns (list[str]): A list of two different string patterns of the timestamp in the {column_to_encode}:
            - The first pattern contains the pattern with a period.
            - The second pattern contains the pattern without the period.

    Returns:
        pd.DataFrame: A modified version of the input DataFrame with the {column_to_encode} encoded to a unix timestamp.
    """
    timestamps = []
    for index, _ in original_dataframe.iterrows():
        time_stamp = original_dataframe.loc[index, column_to_encode]
        if time_stamp.__contains__("."):
            pattern = patterns[0]
        else:
            pattern = patterns[1]
        time_stamp = datetime.strptime(
            original_dataframe.loc[index, column_to_encode], pattern)
        timestamps.append(time_stamp.timestamp())

    original_dataframe[column_to_encode] = timestamps
    return original_dataframe


def encode_utc_timestamp(original_dataframe: pd.DataFrame, column_to_encode: str, pattern: str) -> pd.DataFrame:
    """ 
    Encodes a utc timestamp in a DataFrame column to a unix timestamp, to convert the timestamp into an integer, 
    that can be used as input feature for the neural network.

    Iterates over all entries in the {column_to_encode}, transforms them into an unix timestamp.
    These unix timestamps then get added to a list containing all timestamps. 
    After iterating over all entries the values in the {column_to_encode} get substituted with the list of timestamps. 

    Args:
        original_dataframe (pd.DataFrame): The input DataFrame to encode the utc timestamp.
        column_to_encode (str): The name of the column to modify in the input DataFrame.
        pattern (str): The string pattern of the timestamp in the {column_to_encode}.

    Returns:
        pd.DataFrame: A modified version of the input DataFrame with the {column_to_encode} encoded to a unix timestamp.
    """

    timestamps = []
    for index, _ in original_dataframe.iterrows():
        time_stamp = original_dataframe.loc[index, column_to_encode]
        time_stamp = datetime.strptime(
            original_dataframe.loc[index, column_to_encode], pattern)
        timestamps.append(time_stamp.timestamp())

    original_dataframe[column_to_encode] = timestamps
    return original_dataframe


def encode_boolean(original_dataframe: pd.DataFrame, column_to_encode: str) -> pd.DataFrame:
    """ 
    Transforms the content of column with boolean string representation into a binary integer representation.

    Args:
        original_dataframe (pd.DataFrame): The input DataFrame to modify.
        column_to_encode (str): The name of the column to modify in the input DataFrame.

    Returns:
        pd.DataFrame: A modified version of the input with an integer representation of the boolean variable.
    """
    original_dataframe[column_to_encode] = original_dataframe[column_to_encode].astype(
        'int')
    return original_dataframe


def encode_date(original_dataframe: pd.DataFrame, column_to_encode: str, pattern: str) -> pd.DataFrame:
    """ 
    Transforms a Date as string with the specified string pattern into a ordinal encoded representation.

    Ordinal encoded integers imply an ordering. 

    Args:
        original_dataframe (pd.DataFrame): The input DataFrame to modify.
        column_to_encode (str): The name of the column to modify in the input DataFrame.
        pattern (str): The string pattern of the date string.

    Returns:
        pd.DataFrame: A modified version of the input DataFrame with an ordinal encoded representation of the date variable. 
    """
    for _, row in original_dataframe.iterrows():
        date_object = datetime.strptime(row[column_to_encode], pattern)
        original_dataframe[column_to_encode] = date_object.toordinal()
    return original_dataframe


def encode_weather_enums(original_dataframe: pd.DataFrame, json_subarray_name: str, json_key: str, parsed_json: dict) -> pd.DataFrame:
    """ 
    Transforms the string from the weather_hour data into the in the json defined integer encoded representation of the enum.
    The same representation is also used in the weather_day data. 

    Args:
        original_dataframe (pd.DataFrame): The input DataFrame to modify.
        json_subarray_name (str): The name of the subarray from the json. e.g weather_enums 
        json_key (str): The name of the array inside the subarray defined in the parameter before. e.g. moon_phase
        parsed_json(dict):  A dict containing the parsed Dataset. 

    Returns:
        pd.DataFrame: A modified version of the input DataFrame with integer encoded weather data enums.
    """
    # get the enum data from the json file as dict
    weather_day_data = parsed_json[json_subarray_name][json_key]
    # convert the non-int enum strings to their in the json defined int values
    for index, _ in original_dataframe.iterrows():
        for key, value in weather_day_data.items():
            if original_dataframe.loc[index, f'weather_hour_id {json_key}'] == value:
                original_dataframe.loc[index,
                                       f'weather_hour_id {json_key}'] = key
    return original_dataframe


def encode_weather_data(original_dataframe: pd.DataFrame, weather_id_column: str) -> pd.DataFrame:
    """ 
    Returns the DataFrame with encoded weather data. 

    Args: 
        original_dataframe (pd.DataFrame): The input DataFrame to modify.
        weather_id_column (str):  The name of the weather_data to encode. Either weather_day_id or weather_hour_id.

    Returns:
        pd.DataFrame: A modified version of the input DataFrame with the weather_data encoded. 
    """

    if weather_id_column == "weather_day_id":
        original_dataframe = encode_utc_timestamp(
            original_dataframe, f'{weather_id_column} sun_set', "%Y-%m-%dT%H:%M:%S%z")
        original_dataframe = encode_utc_timestamp(
            original_dataframe, f'{weather_id_column} sun_rise', "%Y-%m-%dT%H:%M:%S%z")
        original_dataframe = encode_utc_timestamp(
            original_dataframe, f'{weather_id_column} created_at', "%Y-%m-%dT%H:%M:%S.%f%z")
        original_dataframe = encode_utc_timestamp(
            original_dataframe, f'{weather_id_column} calculated_at', "%Y-%m-%dT%H:%M:%S%z")

    elif weather_id_column == "weather_hour_id":
        original_dataframe = encode_utc_timestamp(
            original_dataframe, f'{weather_id_column} created_at', "%Y-%m-%dT%H:%M:%S.%f%z")
        original_dataframe = encode_utc_timestamp(
            original_dataframe, f'{weather_id_column} calculated_at', "%Y-%m-%dT%H:%M:%S%z")
        original_dataframe = encode_utc_timestamp(
            original_dataframe, f'{weather_id_column} forecast_time', "%Y-%m-%dT%H:%M:%S%z")

    return original_dataframe


def onehot_encode_column(original_dataframe: pd.DataFrame, column_to_encode: str) -> pd.DataFrame:
    """ 
    One-hot encodes the passed column of the DataFrame. 

    One-hot encoding is a technique used to convert categorical variables into a numerical representation,
    which is necessary to use nominal non-integer features without an inherent order as input in a neural network.
    This guarantees that the neural network can correctly learn from the data without implying any ordinal relationships.

    One-hot encoding splits the specified column by creating new binary columns for each unique value present in the column.
    Each binary column represents if the corresponding value is present for the row in the DataFrame.

    Args:  
        original_dataframe (pd.DataFrame): The input DataFrame to modify via one-hot encoding.
        column_to_encode (str): The name of the column to modify in the input DataFrame.

    Returns:
        pd.DataFrame: A modified version of the input DataFrame with one-hot encoded columns for the {column_to_encode}
        Note: If the {column_to_encode} is not 'session_id', the {column_to_encode} is dropped from the DataFrame after encoding.
        The 'session_id' column is still needed later on when looking for the classification for the y_sets. 
    """
    encoder = OneHotEncoder()
    encoded_column = encoder.fit_transform(
        original_dataframe[[column_to_encode]])
    encoded_column_dataframe = pd.DataFrame(encoded_column.toarray(
    ), columns=encoder.get_feature_names_out([column_to_encode]), index=original_dataframe.index)
    original_dataframe = pd.concat(
        [original_dataframe, encoded_column_dataframe], axis=1)

    if column_to_encode != "session_id":
        original_dataframe = original_dataframe.drop(
            columns=[column_to_encode])

    return original_dataframe


def label_encode_column(original_dataframe: pd.DataFrame, column_to_encode: str) -> pd.DataFrame:
    """ 
    Label encodes the passed column of the DataFrame.

    Label encoding is a technique used to convert categorical variables into a numerical representation,
    which is necessary to use non-integer features as input in a neural network.

    Label encoding does transform the categorical variable into a ordinal representation and therefore implies an order. 

    Args:
        original_dataframe (pd.DataFrame): The input DataFrame to modify via label encoding.
        column_to_encode (str): The name of the column to modify in the input DataFrame.

    Returns:
        pd.DataFrame: A modified version of the input DataFrame with an integer representation of the categorical variable.
    """
    label_encoder = LabelEncoder()
    original_dataframe[column_to_encode] = label_encoder.fit_transform(
        original_dataframe[column_to_encode])
    return original_dataframe


def binary_encode_column(original_dataframe: pd.DataFrame, column_to_encode: str) -> pd.DataFrame:
    """ 
    Binary encodes the passed column of the DataFrame.

    Binary encoding is a technique used to convert categorical variables into a binary representation,
    which is necessary to use non-integer features as input in a neural network.

    Binary encoding first transforms the categorical variables into a numerical representation.
    Afterwards this numerical representation is transformed into its binary number and split into multiple columns. 

    Binary encoding can imply a weak ordering.

    Args:
        original_dataframe (pd.DataFrame): The input DataFrame to modify.
        column_to_encode (str): The name of the column to modify in the input DataFrame.

    Returns:
        pd.DataFrame: A modified version of the input DataFrame with an binary encoded representation of the categorical variable.
    """
    binary_encoder = category_encoders.BinaryEncoder(cols=column_to_encode)
    original_dataframe = binary_encoder.fit_transform(original_dataframe)
    return original_dataframe
