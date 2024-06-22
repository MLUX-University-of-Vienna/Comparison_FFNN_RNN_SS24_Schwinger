"""
This module contains the functions to generate specific plots.

Functions:
    plot_unique_values_from_a_column(original_dataframe: pd.DataFrame, column_name: str) -> None:
        Plots how often each unique value in the column {column_name} of the DataFrame {original_dataframe} appears.

    plot_history(history: list[dict]) -> None:
        Plots the training accuracy and loss over epochs for multiple training processes of a Keras Model.
"""


import matplotlib.pyplot as plt
import pandas as pd


def plot_unique_values_from_a_column(original_dataframe: pd.DataFrame, column_name: str) -> None:
    """
    Plots how often each unique value in the column {column_name} of the DataFrame {original_dataframe} appears.

    Parameters:
        original_dataframe (pd.DataFrame): The DataFrame which unique values in a column should be plotted.
        column_name (str): The name of the column to plot from the DataFrame.

    Returns:
        None
    """
    value_count = original_dataframe[column_name].value_counts()
    plt.bar(value_count.index, value_count.values)
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.title(f"Frequency of each unique value in {column_name}")
    plt.show()


def plot_history(history: list[dict]) -> None:
    """
    Plots the training accuracy and loss over epochs for multiple training processes of a Keras Model.

    Parameters:
        history (list[dict]): A list of dictionary objects returned by the fit() method of a Keras Model. It contains the training metrics recorded during the training process.

    Returns:
        None
    """
    _, axes = plt.subplots(5, 2, figsize=(20, 40))
    for i, history in enumerate(history):
        ax1 = axes[i, 0]
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.set_title(f'Model {i+1} - Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(loc='lower right')

        ax2 = axes[i, 1]
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.set_title(f'Model {i+1} - Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(loc='upper right')

    plt.show()
