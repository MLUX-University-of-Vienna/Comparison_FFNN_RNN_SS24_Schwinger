"""
This module contains the functions split the data for the embedding layers from the remaining, non-embedded features.

Functions:rns a tuple of np.arrays.

    split_3D_set
    split_2D_sets(original_array: np.array) -> tuple[np.array, np.array]:
        Retus(original_array: np.array) -> tuple[np.array, np.array]:
        Returns a tuple of np.arrays.
"""


import numpy as np


def split_2D_sets(original_array: np.array) -> tuple[np.array, np.array]:
    """ 
    Returns a tuple of np.arrays.

    Parameters:
        original_array (np.array): The input train or test array to modify and slice the first column with the data for the embedding layer. 

    Returns:
        tuple[np.array, np.array]: A tuple of two np.arrays containing:
            - A splitted array with the data for the embedding layer.
            - The remaining array with the remaining input features, which are not embedded.        
    """
    return original_array[:, :1], original_array[:, 1:]


def split_3D_sets(original_array: np.array) -> tuple[np.array, np.array]:
    """ 
    Returns a tuple of np.arrays.

    Parameters:
        original_array (np.array): The input train or test array to modify and slice the first column with the data for the embedding layer. 

    Returns:
        tuple[np.array, np.array]: A tuple of two np.arrays containing:
            - A splitted array with the data for the embedding layer.
            - The remaining array with the remaining input features, which are not embedded.        
    """
    return original_array[:, :, :1], original_array[:, :, 1:]
