"""
This module contains the Model superclass.
"""

import keras
import numpy as np

from Helpers import plot


class Model:
    """
    A custom model class for the needed functionality regarding a module and the hypertuning process.


    Attributes:
        X_train_prev_event_input_dim (int): Variable containing the input dimension for the prev_event embedding layer.
        X_train_prev_prev_event_input_dim (int): Variable containing the input dimension for the prev_prev_event embedding layer.
        X_train_session_id_input_dim (int): Variable containing the input dimension for the session_id embedding layer.
        X_train_device_id_input_dim (int): Variable containing the input dimension for the device_id embedding layer.
        X_train (np.array): Array containing the training samples without the data for the embedding layers.
        y_train (np.array): Array containing the labels of the training data.
        unique_classifications (int): Number of unique labels in the dataset.
        X_train_prev_event (np.array): Array containing data to train the prev_event embedding layer.
        X_train_prev_prev_event (np.array): Array containing data to train the prev_prev_event embedding layer.
        X_train_session_id (np.array): Array containing data to train the session_id embedding layer.
        X_train_device_id (np.array): Array containing data to train the device_id embedding layer.
        X_train_content_id (np.array): Array containing data to train the content_id embedding layer.

    Methods:
        visualizeTrainingResults(eval_data: list[tuple[float, float, keras.src.callbacks.history.History]]) -> None:
            Prints the loss and accuracy of each trained model.
            Plots the evolution of the loss and accuracy value for each trained model.
    """

    def __init__(self, X_train_prev_event_input_dim: int, X_train_prev_prev_event_input_dim: int,
                 X_train_session_id_input_dim: int, X_train_device_id_input_dim: int, X_train: np.array, y_train: np.array,
                 unique_classifications: int, X_train_prev_event: np.array, X_train_prev_prev_event: np.array,
                 X_train_session_id: np.array, X_train_device_id: np.array, X_train_content_id: np.array) -> None:
        """
        Initalized a variables of type model with all relevant fields, that will be needed for the rest of the class.

        Args:   
            X_train_prev_event_input_dim (int): Variable containing the input dimension for the prev_event embedding layer.
            X_train_prev_prev_event_input_dim (int): Variable containing the input dimension for the prev_prev_event embedding layer.
            X_train_session_id_input_dim (int): Variable containing the input dimension for the session_id embedding layer.
            X_train_device_id_input_dim (int): Variable containing the input dimension for the device_id embedding layer.
            X_train (np.array): Array containing the training samples without the data for the embedding layers.
            y_train (np.array): Array containing the labels of the training data.
            unique_classifications (int): Number of unique labels in the dataset.
            X_train_prev_event (np.array): Array containing data to train the prev_event embedding layer.
            X_train_prev_prev_event (np.array): Array containing data to train the prev_prev_event embedding layer.
            X_train_session_id (np.array): Array containing data to train the session_id embedding layer.
            X_train_device_id (np.array): Array containing data to train the device_id embedding layer.
            X_train_content_id (np.array): Array containing data to train the content_id embedding layer.

        Returns:
            Model: Initalized Model object.
        """
        self.X_train_prev_event_input_dim = X_train_prev_event_input_dim
        self.X_train_prev_prev_event_input_dim = X_train_prev_prev_event_input_dim
        self.X_train_session_id_input_dim = X_train_session_id_input_dim
        self.X_train_device_id_input_dim = X_train_device_id_input_dim
        self.X_train = X_train
        self.y_train = y_train
        self.unique_classifications = unique_classifications
        self.X_train_prev_event = X_train_prev_event
        self.X_train_prev_prev_event = X_train_prev_prev_event
        self.X_train_session_id = X_train_session_id
        self.X_train_device_id = X_train_device_id
        self.X_train_content_id = X_train_content_id

    def visualizeTrainingResults(self, eval_data: list[tuple[float, float, keras.src.callbacks.history.History]]) -> None:
        """
        Prints the loss and accuracy of each trained model.
        Plots the evolution of the loss and accuracy value for each trained model.

        Args:
            eval_data (list[tuple[float, float, keras.src.callbacks.history.History]]):
                A list containing n-tuples with the loss, accuracy and the history Object of a trained model.
                n is the number of trained models.

        Returns:
            None
        """
        losses = []
        accuracies = []
        histories = []

        for loss, accuracy, history in eval_data:
            losses.append(loss)
            accuracies.append(accuracy)
            histories.append(history)

        print(f'Loss: {losses}\nAccuracy: {accuracies}')
        plot.plot_history(histories)
