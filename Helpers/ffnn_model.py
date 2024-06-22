"""
This module contains the subclass for the feed forward neural network
"""

import keras
import numpy as np
import optuna
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

from Helpers.model import *


class FNN(Model):
    """
    A custom model class for the needed functionality regarding a feed forward neural network and the hypertuning process.


    Attributes:
        Inherits all attributes from the Model superclass

    Methods:
        Inherits visualizeTrainingResults from the Model superclass
        visualizeTrainingResults(eval_data: list[tuple[float, float, keras.src.callbacks.history.History]]) -> None:
            Prints the loss and accuracy of each trained model.
            Plots the evolution of the loss and accuracy value for each trained model.

        create_model(hp_prev: int, hp_prev_prev: int, hp_session: int, hp_device: int, hp_output: int,
                    hp_units: int, hp_layers: int, hp_learning_rate: float, hp_optimizer: str, hp_dropout: float) 
                    -> keras.src.models.functional.Functional:
            Returns a feed forward neural network with the as arguments passed hypertuned parameters.

        objective(trial) -> float:
            Returns the mean value after all crossvalidation folds.
            Searches for the best hyperparameters and then performs a 10-fold cross validation.

        hypertune_model(self, n_trials: int, direction: str) -> optuna.study.study.Study:
            Returns an object with in-dept data for every hyperparameter combination tested in {n_trials}

        train_model(best_trial: optuna.trial.FrozenTrial, X_train_inputs: list[np.array],
                    X_test_inputs: list[np.array], y_test: list[int], epochs: int, batch_size: int) 
                    -> tuple[float, float, keras.src.callbacks.history.History]:
            Returns a tuple with the loss, accuracy and a history object containing the loss and accuracy evolution over the training process.
    """

    def create_model(self, hp_prev: int, hp_prev_prev: int, hp_session: int, hp_device: int, hp_output: int,
                     hp_units: int, hp_layers: int, hp_learning_rate: float, hp_optimizer: str, hp_dropout: float) -> keras.src.models.functional.Functional:
        """ 
        Returns a feed forward neural network with the as arguments passed hypertuned parameters.

        Args:
            hp_prev (int): The hypertuned parameter for the output dimension of the prev_event embedding layer.
            hp_prev_prev (int): The hypertuned parameter for the output dimension of the prev_prev_event embedding layer. 
            hp_session (int): The hypertuned parameter for the output dimension of the session_id embedding layer.
            hp_device(int):  The hypertuned parameter for the output dimension of the device_id embedding layer. 
            hp_output (int): The hypertuned parameter for the output dimension of the output (content_id) embedding layer.
            hp_units (int): The hypertuned parameter for the amount of neurons in the hidden layers.
            hp_layers (int): The hypertuned parameter amount of hidden layers.
            hp_learning_rate (float): The hypertuned parameter for the learning rate.
            hp_optimizer (str): The hypertuned parameter for the optimzier.
            hp_dropout (float): The hypertuned parameter for the dropout rate.

        Returns:
            keras.src.models.functional.Functional: The built model with all the hypertuned parameters.
        """
        # Embedding layers
        embedding_input_prev_event = tf.keras.layers.Input(
            shape=(1,), dtype='int32')
        embedding_layer_prev_event = tf.keras.layers.Embedding(
            input_dim=self.X_train_prev_event_input_dim, output_dim=hp_prev)(embedding_input_prev_event)
        flattened_prev_event = tf.keras.layers.Flatten()(embedding_layer_prev_event)

        embedding_input_prev_prev_event = tf.keras.layers.Input(
            shape=(1,), dtype='int32')
        embedding_layer_prev_prev_event = tf.keras.layers.Embedding(
            input_dim=self.X_train_prev_prev_event_input_dim, output_dim=hp_prev_prev)(embedding_input_prev_prev_event)
        flattened_prev_prev_event = tf.keras.layers.Flatten()(
            embedding_layer_prev_prev_event)

        embedding_input_session_id = tf.keras.layers.Input(
            shape=(1,), dtype='int32')
        embedding_layer_session_id = tf.keras.layers.Embedding(
            input_dim=self.X_train_session_id_input_dim, output_dim=hp_session)(embedding_input_session_id)
        flattened_session_id = tf.keras.layers.Flatten()(embedding_layer_session_id)

        embedding_input_device_id = tf.keras.layers.Input(
            shape=(1,), dtype='int32')
        embedding_layer_device_id = tf.keras.layers.Embedding(
            input_dim=self.X_train_device_id_input_dim, output_dim=hp_device)(embedding_input_device_id)
        flattened_device_id = tf.keras.layers.Flatten()(embedding_layer_device_id)

        # Combine embeddings
        concatenated_embeddings = tf.keras.layers.Concatenate()(
            [flattened_prev_event, flattened_prev_prev_event, flattened_session_id, flattened_device_id])

        # Input for non-embedded features
        flattened_input = tf.keras.layers.Input(shape=(self.X_train.shape[1],))
        flattened = tf.keras.layers.Flatten()(flattened_input)
        concatenated = tf.keras.layers.Concatenate()(
            [concatenated_embeddings, flattened])

        # Add hidden layers
        for _ in range(hp_layers):
            concatenated = tf.keras.layers.Dense(
                units=hp_units, activation='relu')(concatenated)
            concatenated = tf.keras.layers.BatchNormalization()(concatenated)
            concatenated = tf.keras.layers.Dropout(hp_dropout)(concatenated)

        # embedding layer for the output classification variable
        output_embedding_input = tf.keras.layers.Input(
            shape=(1,), dtype='int32')
        output_embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.unique_classifications, output_dim=hp_output)(output_embedding_input)
        flattened_output_embedding = tf.keras.layers.Flatten()(output_embedding_layer)

        # join the whole neural network with the embedding layer for the output
        final_concatenated = tf.keras.layers.Concatenate()(
            [concatenated, flattened_output_embedding])
        output = tf.keras.layers.Dense(
            self.unique_classifications, activation='softmax')(final_concatenated)

        # build the model with all input layers and the output layer
        model = tf.keras.models.Model(inputs=[embedding_input_prev_event, embedding_input_prev_prev_event,
                                              embedding_input_session_id, embedding_input_device_id, flattened_input, output_embedding_input], outputs=output)

        if hp_optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=hp_learning_rate)
        elif hp_optimizer == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=hp_learning_rate)
        elif hp_optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=hp_learning_rate)
        elif hp_optimizer == 'adamw':
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=hp_learning_rate)
        elif hp_optimizer == 'adadelta':
            optimizer = tf.keras.optimizers.Adadelta(
                learning_rate=hp_learning_rate)
        elif hp_optimizer == 'nadam':
            optimizer = tf.keras.optimizers.Nadam(
                learning_rate=hp_learning_rate)
        elif hp_optimizer == 'lion':
            optimizer = tf.keras.optimizers.Lion(
                learning_rate=hp_learning_rate)

        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def objective(self, trial: optuna.trial._trial.Trial) -> float:
        """
            Returns the mean value after all crossvalidation folds.
            Searches for the best hyperparameters and then performs a 10-fold cross validation.

        Args:
            trial (optuna.trial._trial.Trial): Trial object that saves all relevant information about the hyperparameter combination.

        Returns:
            floating: The mean value of calculated over all cross validation folds.
        """

        hp_output1 = trial.suggest_int(
            'output_dim_prev_event', 5, self.X_train_prev_event_input_dim, step=2)
        hp_output2 = trial.suggest_int(
            'output_dim_prev_prev_event', 5, self.X_train_prev_prev_event_input_dim, step=2)
        hp_output3 = trial.suggest_int(
            'output_dim_session_id', 5, self.X_train_session_id_input_dim, step=2)
        hp_output4 = trial.suggest_int(
            'output_dim_device_id', 5, self.X_train_device_id_input_dim, step=2)
        hp_output5 = trial.suggest_int(
            'output_dim_output', 5, self.unique_classifications, step=2)
        hp_units = trial.suggest_int(
            'units', self.unique_classifications, len(self.X_train[0]), step=8)
        hp_layers = trial.suggest_int('layers', 1, 2)
        hp_learning_rate = trial.suggest_float(
            'learning_rate', 0.00001, 0.1, log=True)
        hp_optimizer = trial.suggest_categorical(
            'optimizer', ['adam', 'rmsprop', 'sgd', 'adamw', 'adadelta', 'nadam', 'lion'])
        hp_dropout = trial.suggest_float('dropout', 0.2, 0.5)

        model = self.create_model(hp_output1, hp_output2, hp_output3, hp_output4, hp_output5,
                                  hp_units, hp_layers, hp_learning_rate, hp_optimizer, hp_dropout)

        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        val_accuracies = []

        for train_index, val_index in kfold.split(self.X_train, self.y_train):
            X_train_fold, X_val_fold = self.X_train[train_index], self.X_train[val_index]
            X_train_prev_event_fold, X_val_prev_event_fold = self.X_train_prev_event[
                train_index], self.X_train_prev_event[val_index]
            X_train_prev_prev_event_fold, X_val_prev_prev_event_fold = self.X_train_prev_prev_event[
                train_index], self.X_train_prev_prev_event[val_index]
            X_train_session_fold, X_val_session_fold = self.X_train_session_id[
                train_index], self.X_train_session_id[val_index]
            X_train_device_fold, X_val_device_fold = self. X_train_device_id[
                train_index], self.X_train_device_id[val_index]
            X_train_output_fold, X_val_output_fold = self.X_train_content_id[
                train_index], self.X_train_content_id[val_index]

            X_train_fold_inputs = [X_train_prev_event_fold, X_train_prev_prev_event_fold,
                                   X_train_session_fold, X_train_device_fold, X_train_fold, X_train_output_fold]
            X_val_fold_inputs = [X_val_prev_event_fold, X_val_prev_prev_event_fold,
                                 X_val_session_fold, X_val_device_fold, X_val_fold, X_val_output_fold]

            y_train_fold, y_val_fold = self.y_train[train_index], self.y_train[val_index]

            model.fit(X_train_fold_inputs, y_train_fold, epochs=50, validation_data=(
                X_val_fold_inputs, y_val_fold), verbose=0)

            val_accuracies.append(model.evaluate(
                X_val_fold_inputs, y_val_fold, verbose=0)[1])

        return np.mean(val_accuracies)

    def hypertune_model(self, n_trials: int, direction: str) -> optuna.study.study.Study:
        """
        Returns an object with in-depth data for every hyperparameter combination tested in {n_trials}

        Args:
            n_trials (int): Number, deciding how many possible hyperparameter combinations Optuna should test.
            direction (str): Direction, on which Optuna evaluates the hyperparameter combination.

        Returns:
            optuna.study.study.Study: An object with detailed data for every hyperparamter combination Optuna tested.
        """

        study = optuna.create_study(direction=direction)
        study.optimize(self.objective, n_trials=n_trials)

        print(f'Best hyperparameters: {study.best_params}')

        return study

    def train_model(self, best_trial: optuna.trial.FrozenTrial, X_train_inputs: list[np.array],
                    X_test_inputs: list[np.array], y_test: list[int], epochs: int, batch_size: int) -> tuple[float, float, keras.src.callbacks.history.History]:
        """
        Returns a tuple with the loss, accuracy and a history object containing the loss and accuracy evolution over the training process.

        Args:
            best_trial (optuna.trial.FrozenTrial): Optuna object containing information about the best trial
            X_train_inputs (list[np.array]): A list containing all the np.arrays with the training data for the embedding layers and an np.array with the remaining features.
            X_test_inputs (list[np.array]): A list containing all the np.arrays with the test data for the embedding layers and an np.array with the remaining features.
            y_test (list[int]): A list containing all test labels.
            epochs (int): Number of training of training epochs.
            batch_size (int): Parameter for the batch size.

        Returns:
            tuple[float, float, keras.src.callbacks.history.History]: A Tuple containing:
                - The loss after training the model.
                - The accuracy after training the model.Fs
                - A history object containing the evolution of the loss and accuracy over the training process.
        """
        best_model = self.create_model(best_trial.params['output_dim_prev_event'],
                                       best_trial.params['output_dim_prev_prev_event'],
                                       best_trial.params['output_dim_session_id'],
                                       best_trial.params['output_dim_device_id'],
                                       best_trial.params['output_dim_output'],
                                       best_trial.params['units'],
                                       best_trial.params['layers'],
                                       best_trial.params['learning_rate'],
                                       best_trial.params['optimizer'],
                                       best_trial.params['dropout'])

        early_stopping = keras.callbacks.EarlyStopping(monitor="loss",
                                                       mode="min",
                                                       patience=5)
        history = best_model.fit(X_train_inputs, self.y_train,
                                 epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
        test_loss, test_accuracy = best_model.evaluate(X_test_inputs, y_test)

        print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")
        predictions = best_model.predict(X_test_inputs)
        predicted_classes = np.argmax(predictions, axis=1)
        print(f"Predicted classes: {predicted_classes}")
        return test_loss, test_accuracy, history
