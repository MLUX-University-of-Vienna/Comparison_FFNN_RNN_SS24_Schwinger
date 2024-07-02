import logging
import os

import matplotlib
import numpy as np
from sklearn.model_selection import train_test_split

from Helpers import (load_data, preprocess_dataframe,
                     preprocess_feature_vectors, split_sets)
from Helpers.model import *
from Helpers.rnn_only_content_id import *

matplotlib.use('Qt5Agg')
directory = "logging"
if not os.path.exists(directory):
    os.makedirs(directory)

logger = logging.getLogger(__name__)
log_file_path = os.path.join("logging", "RNN.log")
logging.basicConfig(filename=log_file_path, encoding='utf-8',
                    level=logging.INFO)

dataset = 'demo-s-anon.json'
path_to_dataset = os.path.join('datasets', dataset)
parsed_json = load_data.load_json(path_to_dataset)
all_sessions = load_data.load_data_from_larger_json(parsed_json)

all_sessions = preprocess_dataframe.remove_low_appearance_values(
    all_sessions, 2, 'content_id')

# absolute data with little relevance or data with too many null values
columns_to_drop = preprocess_dataframe.define_columns_to_drop(all_sessions)
columns_to_drop.append('weather_day_id')
columns_to_drop.append('weather_hour_id')

all_sessions = all_sessions.drop(columns=columns_to_drop)

embedded_features = ['content_id', 'session_id']

for column in embedded_features:
    all_sessions = preprocess_dataframe.label_encode_column(
        all_sessions, column)
    all_sessions = preprocess_dataframe.move_column_to_the_front(
        all_sessions, column)

# get the amount of unique entries in the classification column
# + 1 is added for the placeholder that is added afterwards
unique_classifications = all_sessions['content_id'].nunique() + 1

# add (n-1)th and (n-2)th content_id. Move them to front due to the necessary embedding later on and substitute null values.
all_sessions = preprocess_dataframe.addPrevAndPrevPrevEvent(
    all_sessions, unique_classifications, 'content_id', 'session_id')

columns_to_drop = [col for col in all_sessions.columns if col !=
                   'content_id' and col != 'prev_content_id' and col != 'prev_prev_content_id' and col != 'session_id']
all_sessions = all_sessions.drop(columns=columns_to_drop)

grouped_df = all_sessions.groupby(by=["session_id"])

feature_vectors = preprocess_feature_vectors.build_recurrent_feature_vectors(
    grouped_df, 'content_id')

input_features, classification_labels = preprocess_feature_vectors.split_feature_vectors(
    feature_vectors)

sequence_length = preprocess_feature_vectors.calculate_sequence_length(
    input_features)

input_features_padded = preprocess_feature_vectors.pad_input_features(
    input_features, sequence_length)

input_features_padded = np.array(input_features_padded)
classification_labels = np.array(classification_labels)

X_train, X_test, y_train, y_test = train_test_split(
    input_features_padded, classification_labels, test_size=0.1)

number_of_folds = preprocess_feature_vectors.get_number_of_folds(y_test)

# split the X_train and X_test arrays to get the necessary data for the embedding layers
X_train_prev_event, X_train = split_sets.split_3D_sets(X_train)
X_train_prev_prev_event, X_train = split_sets.split_3D_sets(X_train)
X_train_session_id, X_train = split_sets.split_3D_sets(X_train)
X_train_content_id, X_train = split_sets.split_3D_sets(X_train)

X_test_prev_event, X_test = split_sets.split_3D_sets(X_test)
X_test_prev_prev_event, X_test = split_sets.split_3D_sets(X_test)
X_test_session_id, X_test = split_sets.split_3D_sets(X_test)
X_test_content_id, X_test = split_sets.split_3D_sets(X_test)

X_train_prev_event_input_dim = preprocess_feature_vectors.get_embedding_input_dim(
    X_train_prev_event, X_test_prev_event)
X_train_prev_prev_event_input_dim = preprocess_feature_vectors.get_embedding_input_dim(
    X_train_prev_prev_event, X_test_prev_prev_event)
X_train_session_id_input_dim = preprocess_feature_vectors.get_embedding_input_dim(
    X_train_session_id, X_test_session_id)

rnn = RNN_content(X_train_prev_event_input_dim, X_train_prev_prev_event_input_dim, X_train_session_id_input_dim, X_train,
                  y_train, unique_classifications, X_train_prev_event, X_train_prev_prev_event, X_train_session_id, X_train_content_id, number_of_folds)

X_train_inputs = [X_train_prev_event, X_train_prev_prev_event,
                  X_train_session_id, X_train_content_id, X_train]
X_test_inputs = [X_test_prev_event, X_test_prev_prev_event,
                 X_test_session_id, X_test_content_id, X_test]
eval_data = []

for i in range(0, 2):
    study = rnn.hypertune_model(10, 'maximize', True)

    best_trial = study.best_trial

    logging.info(
        f"Only_content: Dataset: {dataset}, Hyperparameters of the best trial for model {i + 1}: {best_trial}")

    for i in range(0, 2):
        eval_data.append(rnn.train_model(best_trial, X_train_inputs,
                                         X_test_inputs, y_test, 200))

rnn.visualizeTrainingResults(eval_data)
