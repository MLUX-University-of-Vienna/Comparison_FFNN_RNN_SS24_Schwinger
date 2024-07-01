import logging

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from Helpers import (load_data, preprocess_dataframe,
                     preprocess_feature_vectors, split_sets)
from Helpers.model import *
from Helpers.rnn_no_previous import *

logger = logging.getLogger(__name__)
logging.basicConfig(filename="logging\\RNN.log",
                    encoding='utf-8', level=logging.INFO)

dataset = 'demo-s-anon.json'
path_to_dataset = f'datasets\\{dataset}'
parsed_json = load_data.load_json(path_to_dataset)
all_sessions = load_data.load_data_from_larger_json(parsed_json)

all_sessions = preprocess_dataframe.remove_low_appearance_values(
    all_sessions, 2, 'content_id')

all_sessions = preprocess_dataframe.replace_null_values_with_column_mode(
    all_sessions)

number_of_folds = preprocess_dataframe.get_number_of_folds(
    all_sessions, 'content_id')

all_sessions = preprocess_dataframe.encode_cyclic_feature(
    all_sessions, 'time_hod', 24)
all_sessions = preprocess_dataframe.encode_cyclic_feature(
    all_sessions, 'time_dow', 7)

all_sessions = preprocess_dataframe.get_weather_data(
    all_sessions, 'weather_day_id', 'weather_day_map', parsed_json)
all_sessions = preprocess_dataframe.get_weather_data(
    all_sessions, 'weather_hour_id', 'weather_hour_map', parsed_json)

all_sessions = preprocess_dataframe.encode_weather_data(
    all_sessions, 'weather_day_id')
all_sessions = preprocess_dataframe.encode_weather_data(
    all_sessions, 'weather_hour_id')

# absolute data with little relevance or data with too many null values
columns_to_drop = preprocess_dataframe.define_columns_to_drop(all_sessions)
all_sessions = all_sessions.drop(columns=columns_to_drop)
all_columns = all_sessions.columns

if any('wind_strength' in columns for columns in all_columns):
    all_sessions = preprocess_dataframe.encode_weather_enums(
        all_sessions, 'weather_enums', 'wind_strength', parsed_json)

if any('wind_direction' in columns for columns in all_columns):
    all_sessions = preprocess_dataframe.encode_weather_enums(
        all_sessions, 'weather_enums', 'wind_direction', parsed_json)
    if ('weather_hour_id wind_direction' in columns for columns in all_columns):
        all_sessions = preprocess_dataframe.encode_cyclic_feature(
            all_sessions, 'weather_hour_id wind_direction', 9)
    elif ('weather_hour_id wind_direction' in columns for columns in all_columns):
        all_sessions = preprocess_dataframe.encode_cyclic_feature(
            all_sessions, 'weather_day_id wind_direction', 9)

if any('thunderstorm_prob' in columns for columns in all_columns):
    all_sessions = preprocess_dataframe.encode_weather_enums(
        all_sessions, 'weather_enums', 'thunderstorm_prob', parsed_json)

if any('moon_phase' in columns for columns in all_columns):
    all_sessions = preprocess_dataframe.encode_cyclic_feature(
        all_sessions, 'weather_day_id moon_phase', 8)

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

# split the X_train and X_test arrays to get the necessary data for the embedding layers
X_train_session_id, X_train = split_sets.split_3D_sets(X_train)
X_train_device_id, X_train = split_sets.split_3D_sets(X_train)
X_train_content_id, X_train = split_sets.split_3D_sets(X_train)

X_test_session_id, X_test = split_sets.split_3D_sets(X_test)
X_test_device_id, X_test = split_sets.split_3D_sets(X_test)
X_test_content_id, X_test = split_sets.split_3D_sets(X_test)

X_train_session_id_input_dim = preprocess_feature_vectors.get_embedding_input_dim(
    X_train_session_id, X_test_session_id)
X_train_device_id_input_dim = preprocess_feature_vectors.get_embedding_input_dim(
    X_train_device_id, X_test_device_id)

rnn = RNN_no_previous(X_train_session_id_input_dim, X_train_device_id_input_dim, X_train, y_train,
                      unique_classifications, X_train_session_id, X_train_device_id, X_train_content_id, number_of_folds)

X_train_inputs = [X_train_session_id,
                  X_train_device_id, X_train_content_id, X_train]
X_test_inputs = [X_test_session_id,
                 X_test_device_id, X_test_content_id, X_test]

eval_data = []

for i in range(0, 2):
    study = rnn.hypertune_model(10, 'maximize', True)

    best_trial = study.best_trial

    logging.info(
        f"No_prev_and_prev_prev Dataset: {dataset}, Hyperparameters of the best trial for model {i + 1}: {best_trial}")

    for i in range(0, 2):
        eval_data.append(rnn.train_model(best_trial, X_train_inputs,
                                         X_test_inputs, y_test, 200))

rnn.visualizeTrainingResults(eval_data)
