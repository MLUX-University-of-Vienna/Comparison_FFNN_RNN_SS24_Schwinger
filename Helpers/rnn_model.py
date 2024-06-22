"""
This module contains the subclass for the recurrent neural network
"""

import keras
import numpy as np
import optuna
import tensorflow as tf
from sklearn.model_selection import KFold, StratifiedKFold

from Helpers.model import *


class RNN(Model):
    """
    A custom model class for the needed functionality regarding a recurrent network and the hypertuning process.


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
                - The accuracy after training the model.
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
                                       best_trial.params['dropout'],
                                       best_trial.params['recurrent_dropout'])

        early_stopping = keras.callbacks.EarlyStopping(monitor="loss",
                                                       mode="min",
                                                       patience=5)
        history = best_model.fit(X_train_inputs, self.y_train,
                                 epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
        test_loss, test_accuracy = best_model.evaluate(X_test_inputs, y_test)
        print("Y_Test: ", y_test)
        print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")
        predictions = best_model.predict(X_test_inputs)
        predicted_classes = np.argmax(predictions, axis=1)
        print(f"Predicted classes: {predicted_classes}")
        return test_loss, test_accuracy, history

    def create_model(self, hp_prev: int, hp_prev_prev: int, hp_session: int, hp_device: int, hp_output: int,
                     hp_units: int, hp_layers: int, hp_learning_rate: float, hp_optimizer: str, hp_dropout: float, hp_recurrent_dropout: float) -> keras.src.models.functional.Functional:
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
            hp_recurrent_dropout (float): The hypertuned parameter for the recurrent dropout

        Returns:
            keras.src.models.functional.Functional: The built model with all the hypertuned parameters.
        """

        embedding_input_prev_event = tf.keras.layers.Input(
            shape=(self.X_train_prev_event.shape[1],))
        embedding_layer_prev_event = tf.keras.layers.Embedding(
            input_dim=2, output_dim=hp_prev)(embedding_input_prev_event)

        embedding_input_prev_prev_event = tf.keras.layers.Input(
            shape=(self.X_train_prev_prev_event.shape[1],))
        embedding_layer_prev_prev_event = tf.keras.layers.Embedding(
            input_dim=self.X_train_prev_prev_event_input_dim, output_dim=hp_prev_prev)(embedding_input_prev_prev_event)

        embedding_input_session_id = tf.keras.layers.Input(
            shape=(self.X_train_session_id.shape[1],))
        embedding_layer_session_id = tf.keras.layers.Embedding(
            input_dim=self.X_train_session_id_input_dim, output_dim=hp_session)(embedding_input_session_id)

        embedding_input_device_id = tf.keras.layers.Input(
            shape=(self.X_train_device_id.shape[1],))
        embedding_layer_device_id = tf.keras.layers.Embedding(
            input_dim=self.X_train_device_id_input_dim, output_dim=hp_device)(embedding_input_device_id)

        embedding_input_content_id = tf.keras.layers.Input(
            shape=(self.X_train_content_id.shape[1],))
        embedding_layer_content_id = tf.keras.layers.Embedding(
            input_dim=self.unique_classifications, output_dim=hp_output)(embedding_input_content_id)

        input_features = tf.keras.layers.Input(
            shape=(self.X_train.shape[1], self.X_train.shape[2]))

        concatenated_embeddings = tf.keras.layers.Concatenate(
            axis=-1)([embedding_layer_prev_event, embedding_layer_prev_prev_event, embedding_layer_session_id, embedding_layer_device_id, embedding_layer_content_id, input_features])

        for i in range(hp_layers):
            if i + 1 != hp_layers:
                concatenated_embeddings = tf.keras.layers.SimpleRNN(
                    units=hp_units,
                    activation='relu',
                    return_sequences=True,
                    dropout=hp_dropout,
                    recurrent_dropout=hp_recurrent_dropout)(concatenated_embeddings)
            else:
                concatenated_embeddings = tf.keras.layers.SimpleRNN(
                    units=hp_units,
                    activation='relu',
                    return_sequences=False,
                    dropout=hp_dropout,
                    recurrent_dropout=hp_recurrent_dropout)(concatenated_embeddings)

        outputs = tf.keras.layers.Dense(
            self.unique_classifications, activation='softmax')(concatenated_embeddings)

        model = tf.keras.models.Model(inputs=[embedding_input_prev_event, embedding_input_prev_prev_event,
                                      embedding_input_session_id, embedding_input_device_id, embedding_input_content_id, input_features], outputs=outputs)

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

    def objective_kFold(self, trial: optuna.trial._trial.Trial) -> float:
        units_upperbound = len(self.X_train)
        if units_upperbound < self.unique_classifications:
            units_upperbound += self.unique_classifications // 2

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
            'units', self.unique_classifications, units_upperbound, step=8)
        hp_layers = trial.suggest_int('layers', 1, 2)
        hp_learning_rate = trial.suggest_float(
            'learning_rate', 0.00001, 0.1, log=True)
        hp_optimizer = trial.suggest_categorical(
            'optimizer', ['adam', 'rmsprop', 'sgd', 'adamw', 'adadelta', 'nadam', 'lion'])
        hp_dropout = trial.suggest_float('dropout', 0.2, 0.5)
        hp_recurrent_dropout = trial.suggest_float(
            'recurrent_dropout', 0.2, 0.5)

        model = self.create_model(hp_output1, hp_output2, hp_output3, hp_output4, hp_output5,
                                  hp_units, hp_layers, hp_learning_rate, hp_optimizer, hp_dropout, hp_recurrent_dropout)

        kfold = KFold(n_splits=10, shuffle=True)
        val_accuracies = []

        for train_index, val_index in kfold.split(self.X_train):
            X_train_fold, X_val_fold = self.X_train[train_index], self.X_train[val_index]
            X_train_prev_event_fold, X_val_prev_event_fold = self.X_train_prev_event[
                train_index], self.X_train_prev_event[val_index]
            X_train_prev_prev_event_fold, X_val_prev_prev_event_fold = self.X_train_prev_prev_event[
                train_index], self.X_train_prev_prev_event[val_index]
            X_train_session_fold, X_val_session_fold = self.X_train_session_id[
                train_index], self.X_train_session_id[val_index]
            X_train_device_fold, X_val_device_fold = self.X_train_device_id[
                train_index], self.X_train_device_id[val_index]
            X_train_output_fold, X_val_output_fold = self.X_train_content_id[
                train_index], self.X_train_content_id[val_index]

            X_train_fold_inputs = [X_train_prev_event_fold, X_train_prev_prev_event_fold,
                                   X_train_session_fold, X_train_device_fold, X_train_output_fold, X_train_fold]
            X_val_fold_inputs = [X_val_prev_event_fold, X_val_prev_prev_event_fold,
                                 X_val_session_fold, X_val_device_fold, X_val_output_fold, X_val_fold]

            y_train_fold, y_val_fold = self.y_train[train_index], self.y_train[val_index]

            model.fit(X_train_fold_inputs, y_train_fold, epochs=50, validation_data=(
                X_val_fold_inputs, y_val_fold), verbose=0)

            val_accuracies.append(model.evaluate(
                X_val_fold_inputs, y_val_fold, verbose=0)[1])

        return np.mean(val_accuracies)

    def objective_strat_kFold(self, trial: optuna.trial._trial.Trial) -> float:
        units_upperbound = len(self.X_train)
        if units_upperbound < self.unique_classifications:
            units_upperbound += self.unique_classifications // 2

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
            'units', self.unique_classifications, units_upperbound, step=8)
        hp_layers = trial.suggest_int('layers', 1, 2)
        hp_learning_rate = trial.suggest_float(
            'learning_rate', 0.00001, 0.1, log=True)
        hp_optimizer = trial.suggest_categorical(
            'optimizer', ['adam', 'rmsprop', 'sgd', 'adamw', 'adadelta', 'nadam', 'lion'])
        hp_dropout = trial.suggest_float('dropout', 0.2, 0.5)
        hp_recurrent_dropout = trial.suggest_float(
            'recurrent_dropout', 0.2, 0.5)

        model = self.create_model(hp_output1, hp_output2, hp_output3, hp_output4, hp_output5,
                                  hp_units, hp_layers, hp_learning_rate, hp_optimizer, hp_dropout, hp_recurrent_dropout)

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
            X_train_device_fold, X_val_device_fold = self.X_train_device_id[
                train_index], self.X_train_device_id[val_index]
            X_train_output_fold, X_val_output_fold = self.X_train_content_id[
                train_index], self.X_train_content_id[val_index]

            X_train_fold_inputs = [X_train_prev_event_fold, X_train_prev_prev_event_fold,
                                   X_train_session_fold, X_train_device_fold, X_train_output_fold, X_train_fold]
            X_val_fold_inputs = [X_val_prev_event_fold, X_val_prev_prev_event_fold,
                                 X_val_session_fold, X_val_device_fold, X_val_output_fold, X_val_fold]

            y_train_fold, y_val_fold = self.y_train[train_index], self.y_train[val_index]

            model.fit(X_train_fold_inputs, y_train_fold, epochs=50, validation_data=(
                X_val_fold_inputs, y_val_fold), verbose=0)

            val_accuracies.append(model.evaluate(
                X_val_fold_inputs, y_val_fold, verbose=0)[1])

        return np.mean(val_accuracies)

    def hypertune_model(self, n_trials: int, direction: str, strat_KFold: bool) -> optuna.study.study.Study:
        """
        Returns an object with in-depth data for every hyperparameter combination tested in {n_trials}

        Args:
            n_trials (int): Number, deciding how many possible hyperparameter combinations Optuna should test.
            direction (str): Direction, on which Optuna evaluates the hyperparameter combination.
            start_KFold (bool): Boolean that determines if stratified K_fold should be used or normal K_fold.

        Returns:
            optuna.study.study.Study: An object with detailed data for every hyperparamter combination Optuna tested.
        """
        if strat_KFold == True:
            study = optuna.create_study(direction=direction)
            study.optimize(self.objective_strat_kFold, n_trials=n_trials)
        else:
            study = optuna.create_study(direction=direction)
            study.optimize(self.objective_kFold, n_trials=n_trials)

        print(f'Best hyperparameters: {study.best_params}')

        return study
