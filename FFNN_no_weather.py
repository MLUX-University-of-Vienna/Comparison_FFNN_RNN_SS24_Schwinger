import logging
import os

import matplotlib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from Helpers import (load_data, preprocess_dataframe,
                     preprocess_feature_vectors, split_sets)
from Helpers.ffnn_model import *
from Helpers.model import *

matplotlib.use('Qt5Agg')
directory = "logging"
if not os.path.exists(directory):
    os.makedirs(directory)

logger = logging.getLogger(__name__)
log_file_path = os.path.join("logging", "FFNN.log")
logging.basicConfig(filename=log_file_path, encoding='utf-8',
                    level=logging.INFO)

dataset = 'demo-s-anon.json'
path_to_dataset = os.path.join('datasets', dataset)
parsed_json = load_data.load_json(path_to_dataset)
all_sessions = load_data.load_data_from_larger_json(parsed_json)

all_sessions = preprocess_dataframe.remove_low_appearance_values(
    all_sessions, 2, 'content_id')

all_sessions = preprocess_dataframe.replace_null_values_with_column_mode(
    all_sessions)

all_sessions = preprocess_dataframe.encode_cyclic_feature(
    all_sessions, 'time_hod', 24)
all_sessions = preprocess_dataframe.encode_cyclic_feature(
    all_sessions, 'time_dow', 7)

# absolute data with little relevance or data with too many null values
columns_to_drop = preprocess_dataframe.define_columns_to_drop(all_sessions)
columns_to_drop.append('weather_day_id')
columns_to_drop.append('weather_hour_id')

all_sessions = all_sessions.drop(columns=columns_to_drop)

ohe_features = preprocess_dataframe.define_ohe_features_list(all_sessions)

ordinal_features = preprocess_dataframe.define_ordinal_features_list(
    all_sessions)

embedded_features = ['content_id', 'device_id', 'session_id']

for column in embedded_features:
    all_sessions = preprocess_dataframe.label_encode_column(
        all_sessions, column)
    all_sessions = preprocess_dataframe.move_column_to_the_front(
        all_sessions, column)

scaler = StandardScaler()
for column in ordinal_features:
    all_sessions[column] = scaler.fit_transform(
        np.array(all_sessions[column]).reshape(-1, 1))

one_hot_encoder = OneHotEncoder()
for column in ohe_features:
    all_sessions = preprocess_dataframe.onehot_encode_column(
        all_sessions, column)

# get the amount of unique entries in the classification column
# + 1 is added for the placeholder that is added afterwards
unique_classifications = all_sessions['content_id'].nunique() + 1

# add (n-1)th and (n-2)th content_id. Move them to front due to the necessary embedding later on and substitute null values.
all_sessions = preprocess_dataframe.addPrevAndPrevPrevEvent(
    all_sessions, unique_classifications, 'content_id', 'session_id')

feature_vectors = preprocess_feature_vectors.build_feed_forward_feature_vectors(
    all_sessions, 'content_id')

input_features, classification_labels = preprocess_feature_vectors.split_feature_vectors(
    feature_vectors)

# transform the lists to np.arrays to build train / test sets
input_features_array = np.array(input_features)
classification_labels_array = np.array(classification_labels)

X_train, X_test, y_train, y_test = train_test_split(
    input_features_array, classification_labels_array, test_size=0.1)

number_of_folds = preprocess_feature_vectors.get_number_of_folds(y_test)

# split the X_train and X_test arrays to get the necessary data for the embedding layers
X_train_prev_event, X_train = split_sets.split_2D_sets(X_train)
X_train_prev_prev_event, X_train = split_sets.split_2D_sets(X_train)
X_train_session_id, X_train = split_sets.split_2D_sets(X_train)
X_train_device_id, X_train = split_sets.split_2D_sets(X_train)
X_train_content_id, X_train = split_sets.split_2D_sets(X_train)

X_test_prev_event, X_test = split_sets.split_2D_sets(X_test)
X_test_prev_prev_event, X_test = split_sets.split_2D_sets(X_test)
X_test_session_id, X_test = split_sets.split_2D_sets(X_test)
X_test_device_id, X_test = split_sets.split_2D_sets(X_test)
X_test_content_id, X_test = split_sets.split_2D_sets(X_test)

X_train_prev_event_input_dim = preprocess_feature_vectors.get_embedding_input_dim(
    X_train_prev_event, X_test_prev_event)
X_train_prev_prev_event_input_dim = preprocess_feature_vectors.get_embedding_input_dim(
    X_train_prev_prev_event, X_test_prev_prev_event)
X_train_session_id_input_dim = preprocess_feature_vectors.get_embedding_input_dim(
    X_train_session_id, X_test_session_id)
X_train_device_id_input_dim = preprocess_feature_vectors.get_embedding_input_dim(
    X_train_device_id, X_test_device_id)

ffnn = FNN(X_train_prev_event_input_dim, X_train_prev_prev_event_input_dim, X_train_session_id_input_dim, X_train_device_id_input_dim, X_train,
           y_train, unique_classifications, X_train_prev_event, X_train_prev_prev_event, X_train_session_id, X_train_device_id, X_train_content_id, number_of_folds)

X_train_inputs = [X_train_prev_event, X_train_prev_prev_event,
                  X_train_session_id, X_train_device_id, X_train, X_train_content_id]
X_test_inputs = [X_test_prev_event, X_test_prev_prev_event,
                 X_test_session_id, X_test_device_id, X_test, X_test_content_id]
eval_data = []

for i in range(0, 2):
    study = ffnn.hypertune_model(
        n_trials=10, direction='maximize')

    best_trial = study.best_trial

    logging.info(
        f"No_weather: Dataset: {dataset}, Hyperparameters of the best trial for model {i + 1}: {best_trial}")

    for i in range(0, 2):
        eval_data.append(ffnn.train_model(best_trial, X_train_inputs,
                                          X_test_inputs, y_test, 200))

ffnn.visualizeTrainingResults(eval_data)
