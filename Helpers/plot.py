"""
This module contains the functions to generate specific plots.

Functions:
    plot_unique_values_from_a_column(original_dataframe: pd.DataFrame, column_name: str) -> None:
        Plots how often each unique value in the column {column_name} of the DataFrame {original_dataframe} appears.

    plot_history(history: list[dict]) -> None:
        Plots the training accuracy and loss over epochs for multiple training processes of a Keras Model.

    plot_accuracy_loss(accuracies: list[int], losses: list[int]) -> None:
        Plots the final accuracy and loss for multiple evaluation processes of multiple Keras Models.
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
    Plots the training accuracy and loss over epochs for multiple training processes of multiple Keras Models.

    Parameters:
        history (list[dict]): A list of dictionary objects returned by the fit() method of a Keras Model. It contains the training metrics recorded during the training process.

    Returns:
        None
    """
    _, axes = plt.subplots(1, 2, figsize=(20, 5))

    ax1 = axes[0]
    ax1.set_title("Accuracy")
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')

    ax2 = axes[1]
    ax2.set_title("Loss")
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')

    colors = ['purple', 'cyan', 'orange']
    linestyle = ""

    for i, hist in enumerate(history):
        color = colors[i // 2]
        if i % 2 == 0:
            linestyle = "-"
        else:
            linestyle = "--"

        ax1.plot(hist.history['accuracy'], color=color, linestyle=linestyle,
                 label=f'Training Accuracy: {i + 1}')
        ax2.plot(hist.history['loss'], color=color, linestyle=linestyle,
                 label=f'Training Loss: {i + 1}')

    ax1.legend(loc='lower right')
    ax2.legend(loc='upper right')

    plt.show()


def plot_accuracy_loss(accuracies: list[int], losses: list[int]) -> None:
    """ 
    Plots the final accuracy and loss for multiple evaluation processes of multiple Keras Models.

    Parameters:
        accuracies (list[int]): A list containing the final accuracies of each evaluation run.
        losses (list[int]): A list containing the final losses of each evaluation run.

    Returns:
        None
    """
    runs = range(1, len(accuracies) + 1)

    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_loss = sum(losses) / len(losses)

    _, axes = plt.subplots(1, 2, figsize=(20, 5))
    plt.title('Accuracy and Loss Across Different Runs')

    axes[0].set_title("Accuracy")
    axes[1].set_title("Loss")

    axes[0].plot(runs, accuracies, marker='o', linestyle='-',
                 color='purple', label='Accuracy')
    axes[0].axhline(avg_accuracy, color='turquoise', linestyle='--',
                    label=f'Average Accuracy: {avg_accuracy:.2f}')
    axes[0].set_xlabel('Runs')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(loc='upper left')

    axes[1].plot(runs, losses, marker='o', linestyle='-',
                 color='purple', label='Loss')
    axes[1].axhline(avg_loss, color='turquoise', linestyle='--',
                    label=f'Average Loss: {avg_loss:.2f}')
    axes[1].set_xlabel('Runs')
    axes[1].set_ylabel('Loss')
    axes[1].legend(loc='upper left')

    plt.show()
