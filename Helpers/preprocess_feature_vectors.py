"""
This module contains the functions to build and get the feature vectors.
It also contains a method to get the input dimension for the embedding layers.

Functions:
    build_recurrent_feature_vectors(grouped_df: pd.DataFrame, classification_column_name: str) -> list[list[tuple[pd.Series, float]]]:
        Returns the input_feature_vectors for the recurrent neural network and their classifications in form of a list containing a sublist 
        for each session with a tuple containing the input feature vector as pd.Series and the classification as float.

    build_feed_forward_feature_vectors(original_dataframe: pd.DataFrame, classification_column_name: str) -> list[tuple[pd.Series, float]]:
        Returns the input_feature_vectors for the recurrent neural network and their classifications in form of a list containing 
        a tuple containing with the input feature vector as pd.Series and the classification as float.

    split_feature_vectors(feature_vectors: list[tuple[pd.Series, float]]) -> tuple[list[pd.Series], list[float]]:
        Splits and returns the input_feature_vector tuples from the feed forward network into two lists. 
        The first containing the input_feature_vectors.
        The other containing the classifications for the input_feature_vectors

    calculate_sequence_length(input_features: list[list[pd.Series]]) -> int:
        Returns the median sequence length calculated with the help of all sequences.
        The sequence length is rounded up to the next integer value. 

    pad_input_features(input_features: list[list[float]], sequence_length: int) -> list[list[float]]:
        Returns a multidimensional list of padded input_features.
    
    pad_classifications(list_of_all_classifications: list[float], sequence_length: int) -> list[float]:
        Return a multidimensional list of padded classifications.

    get_embedding_input_dim(train_array_to_embed: np.array, test_array_to_embed: np.array) -> int:
        Returns the maximum value of the train and test arrays to calculate the input dimension for the dimension of the embedding layer.
"""


import math
import statistics

import numpy as np
import pandas as pd
from scipy import stats

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def build_recurrent_feature_vectors(grouped_df: pd.DataFrame, classification_column_name: str) -> list[list[tuple[pd.Series, float]]]:
    """
    Returns the input_feature_vectors for the recurrent neural network and their classifications in form of a list containing a sublist 
    for each session with a tuple containing the input feature vector as pd.Series and the classification as float.

    Args:
        grouped_df (df.DataFrame): The grouped input dataframe to build the feature_vectors.
        classification_column_name (str): The name of the column of the {grouped_df} that contains the classification. 

    Returns:
        list[list[tuple[pd.Series, float]]]: A list of sublist containing tuples of a pd.Series and a float value:
            - The input feature vector as pd.Series from the {grouped_df}.
            - The classification of the input feature vector as float value.
    """
    feature_vectors = []

    for _, group in grouped_df:
        feature_vector = []
        classification = 0
        for index in range(0, len(group)):
            if index < len(group) - 1:  # remove last sample of session
                feature_vector.append((group.iloc[index]))
            if index == len(group) - 1:
                classification = group.iloc[index][classification_column_name]
        if (len(feature_vector) > 0):
            feature_vectors.append((feature_vector, classification))

    return feature_vectors


def build_feed_forward_feature_vectors(original_dataframe: pd.DataFrame, classification_column_name: str) -> list[tuple[pd.Series, float]]:
    """
    Returns the input_feature_vectors for the recurrent neural network and their classifications in form of a list containing 
    a tuple containing with the input feature vector as pd.Series and the classification as float.

    Args:
        original_dataframe (df.DataFrame): The input dataframe to build the feature_vectors.
        classification_column_name (str): The name of the column of the {original_dataframe} that contains the classification. 

    Returns:
        list[tuple[pd.Series, float]]: A list containing tuples of a pd.Series and a float value:
            - The input feature vector as pd.Series from the {original_dataframe}.
            - The classification of the input feature vector as float value.
    """
    feature_vectors = []
    for i in range(0, len(original_dataframe)):
        if i < len(original_dataframe) - 1:
            if original_dataframe.iloc[i]['session_id'] == original_dataframe.iloc[i+1]['session_id']:
                feature_vector = (
                    original_dataframe.iloc[i], original_dataframe.iloc[i+1][classification_column_name])
                feature_vectors.append(feature_vector)
    return feature_vectors


def split_feature_vectors(feature_vectors: list[tuple[pd.Series, float]]) -> tuple[list[pd.Series], list[float]]:
    """ 
    Splits and returns the input_feature_vector tuples from the feed forward network into two lists. 
    The first containing the input_feature_vectors.
    The other containing the classifications for the input_feature_vectors

    Args:
        feature_vectors (list[tuple[pd.Series, float]]): A list containing all sessions and their classification.

    Returns:
        tuple[list[pd.Series], list[float]]: A tuple of two list containing:
            - A list containing the input_feature_vectors as pd.Series. 
            - A list containing the classifications for the input_feature_vectors as float values.
    """
    input_features = []
    classification_labels = []

    for input_feature, classification in feature_vectors:
        input_features.append(input_feature)
        classification_labels.append(classification)

    return input_features, classification_labels


def calculate_sequence_length(input_features: list[list[pd.Series]]) -> int:
    """
    Returns the median sequence length calculated with the help of all sequences.
    The sequence length is rounded up to the next integer value.

    Args:
        input_features: list[list[pd.Series]]: A list containing all sessions.

    Returns:
        int: The sequence length rounded up 
    """
    all_sequence_lengths = [len(seq) for seq in input_features]
    return math.ceil(statistics.median(all_sequence_lengths))


def pad_input_features(input_features: list[list[float]], sequence_length: int) -> list[list[float]]:
    """ 
    Returns a multidimensional list of padded input_features.

    Args: 
        input_features (list[list[float]]): A list containing all input features.
        sequence_length (int): The previously calculated sequence_length for every feature in every subsession. 

    Returns:
        list[list[float]]: A list of sessions, where each subsession is padded to the desired length.
        Note: All existing entries of a session get converted to list[float] from pd.Series.
    """
    padded_list = []
    for subsession in input_features:
        # Calculate the mode of the current session based on the first session entry
        # [0] returns the value, [1] return the count how often the values appears in the list
        mode_value = list(stats.mode(subsession)[0])

        # convert all pd.Series in the subsession entries to lists
        for i in range(len(subsession)):
            subsession[i] = subsession[i].tolist()

        # Check if the subsession needs to be padded
        if len(subsession) < sequence_length:
            range_to_pad = sequence_length - len(subsession)
            for _ in range(range_to_pad):
                subsession.insert(0, mode_value)
        else:
            subsession = subsession[:sequence_length]

        padded_list.append(subsession)
    return padded_list


def pad_classifications(list_of_all_classifications: list[float], sequence_length: int) -> list[float]:
    """ 
    Return a multidimensional list of padded classifications.

    Args: 
        list_of_all_classifications (list[float]): A list containing all classifications.
        sequence_length (int): The previously calculated sequence_length for every subsession.

    Returns:
        list[float]: A list of sessions, where each subsession is padded to the desired length.
        Note: All existing entries of a session get converted to list[float] from pd.Series.
    """
    padded_list = []
    for classification_for_subsession in list_of_all_classifications:
        # Calculate the mode of the classification for the current subsession
        # [0] returns the value, [1] return the count how often the values appears in the list
        mode_value = stats.mode(classification_for_subsession)[0]

        # Check if the subsession needs to be padded
        if len(classification_for_subsession) < sequence_length:
            range_to_pad = sequence_length - len(classification_for_subsession)
            for _ in range(range_to_pad):
                classification_for_subsession.insert(0, mode_value)
        else:
            classification_for_subsession = classification_for_subsession[:sequence_length]

        padded_list.append(classification_for_subsession)
    return padded_list


def get_embedding_input_dim(train_array_to_embed: np.array, test_array_to_embed: np.array) -> int:
    """ 
    Returns the maximum value of the train and test arrays to calculate the input dimension for the dimension of the embedding layer.

    Parameters:
        train_array_to_embed (np.array): The array with the training data used..
        test_array_to_embed (np.array): The array with the test data used.

    Returns:
        int: The necessary size of the input dimension for the embedding layer.
    """
    train_max = int(max(set(train_array_to_embed.flatten()))) + 1
    test_max = int(max(set(test_array_to_embed.flatten()))) + 1
    return max(train_max, test_max)
