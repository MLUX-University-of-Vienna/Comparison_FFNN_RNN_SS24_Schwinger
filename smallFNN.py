import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from Helpers import (load_data, preprocess_dataframe,
                     preprocess_feature_vectors, split_sets)
from Helpers.ffnn_model import *
from Helpers.model import *

path_to_dataset = "datasets\\transfer\\smaller_dataset.json"
parsed_json = load_data.load_json(path_to_dataset)
all_sessions = load_data.load_data_from_smaller_json(parsed_json)

all_sessions = preprocess_dataframe.remove_low_appearance_values(
    all_sessions, 2, 'content_id')

all_sessions = all_sessions.drop(
    columns=['weather_future_day_id', 'weather_future_hour_id'])

all_sessions = preprocess_dataframe.replace_null_values_with_column_mode(
    all_sessions)

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
columns_to_drop = ['weather_day_id moon_set', 'weather_day_id moon_rise', 'content_portal', 'device_online', 'time_utc', 'time_local',
                   'device_height_px', 'device_width_px', 'weather_day_id sun_set', 'weather_day_id sun_rise', 'weather_day_id moon_set',
                   'weather_day_id moon_rise', 'weather_day_id created_at', 'weather_day_id calculated_at', 'weather_day_id forecast_date',
                   'weather_hour_id created_at', 'weather_hour_id calculated_at', 'weather_hour_id forecast_time']

all_sessions = all_sessions.drop(columns=columns_to_drop)

all_sessions = preprocess_dataframe.encode_weather_enums(
    all_sessions, 'weather_enums', 'wind_strength', parsed_json)
all_sessions = preprocess_dataframe.encode_weather_enums(
    all_sessions, 'weather_enums', 'wind_direction', parsed_json)
all_sessions = preprocess_dataframe.encode_weather_enums(
    all_sessions, 'weather_enums', 'thunderstorm_prob', parsed_json)

all_sessions = preprocess_dataframe.encode_cyclic_feature(
    all_sessions, 'weather_hour_id wind_direction', 9)
all_sessions = preprocess_dataframe.encode_cyclic_feature(
    all_sessions, 'weather_day_id wind_direction', 9)
all_sessions = preprocess_dataframe.encode_cyclic_feature(
    all_sessions, 'weather_day_id moon_phase', 8)


ohe_features = ['device_class', 'device_orientation', 'oha_language_iso2', 'oha_layout',
                'device_country_iso2', 'device_language_iso2', 'event_type', 'device_platform']

ordinal_features = ['time_dow_sin', 'time_dow_cos', 'time_hod_sin', 'time_hod_cos', 'weather_hour_id thunderstorm_prob', 'weather_day_id thunderstorm_prob', 'weather_hour_id wind_direction_sin',
                    'weather_hour_id wind_direction_cos', 'weather_day_id sunshine_h',
                    'weather_day_id temp_max_c', 'weather_day_id temp_min_c', 'weather_day_id moon_phase_sin',
                    'weather_day_id prec_prob_pct', 'weather_day_id prec_rain_mm_h', 'weather_day_id prec_snow_mm_h', 'weather_day_id wind_speed_kmh',
                    'weather_day_id prec_total_mm_h', 'weather_day_id temp_felt_max_c', 'weather_day_id temp_felt_min_c', 'weather_day_id humidity_mean_pct',
                    'weather_day_id wind_speed_max_kmh', 'weather_day_id cloud_cover_max_pct', 'weather_day_id cloud_cover_min_pct', 'weather_day_id cloud_cover_mean_pct',
                    'weather_hour_id temp_c', 'weather_hour_id sunshine_h', 'weather_hour_id temp_felt_c', 'weather_hour_id humidity_pct',
                    'weather_hour_id prec_rain_mm_h', 'weather_hour_id prec_snow_mm_h', 'weather_day_id wind_direction_sin',
                    'weather_day_id wind_direction_cos', 'weather_hour_id wind_speed_kmh', 'weather_hour_id cloud_cover_pct', 'weather_hour_id prec_total_mm_h',
                    'weather_hour_id forecast_distance_h', 'weather_hour_id wind_strength', 'weather_day_id wind_strength', 'weather_day_id moon_phase_cos']

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
# TODO: fix
all_sessions[f'prev_content_id'] = all_sessions['content_id'].shift(1)
all_sessions[f'prev_prev_content_id'] = all_sessions['content_id'].shift(2)
all_sessions = preprocess_dataframe.move_column_to_the_front(
    all_sessions, 'prev_prev_content_id')
all_sessions = preprocess_dataframe.move_column_to_the_front(
    all_sessions, 'prev_content_id')
all_sessions.fillna(
    {'prev_content_id': unique_classifications}, inplace=True)
all_sessions.fillna(
    {'prev_prev_content_id': unique_classifications}, inplace=True)

feature_vectors = preprocess_feature_vectors.build_feed_forward_feature_vectors(
    all_sessions, 'content_id')

input_features, classification_labels = preprocess_feature_vectors.split_feature_vectors(
    feature_vectors)

# transform the lists to np.arrays to build train / test sets
input_features_array = np.array(input_features)
classification_labels_array = np.array(classification_labels)

X_train, X_test, y_train, y_test = train_test_split(
    input_features_array, classification_labels_array, test_size=0.1)

# split the X_train and X_test arrays to get the necessary data for the embedding layers
X_train_prev_event, X_train = split_sets.split_2D_sets(X_train)
X_train_prev_prev_event, X_train = split_sets.split_2D_sets(X_train)
X_train_session_id, X_train = split_sets.split_2D_sets(X_train)
X_train_device_id, X_train = split_sets.split_2D_sets(X_train)
X_train_content_id, X_train = split_sets.split_2D_sets(X_train)

X_test_prev_prev_event, X_test = split_sets.split_2D_sets(X_test)
X_test_prev_event, X_test = split_sets.split_2D_sets(X_test)
X_test_session_id, X_test = split_sets.split_2D_sets(X_test)
X_test_device_id, X_test = split_sets.split_2D_sets(X_test)
X_test_content_id, X_test = split_sets.split_2D_sets(X_test)

X_train_prev_event_input_dim = preprocess_feature_vectors.get_embedding_input_dim(
    X_train_prev_event)
X_train_prev_prev_event_input_dim = preprocess_feature_vectors.get_embedding_input_dim(
    X_train_prev_prev_event)
X_train_session_id_input_dim = preprocess_feature_vectors.get_embedding_input_dim(
    X_train_session_id)
X_train_device_id_input_dim = preprocess_feature_vectors.get_embedding_input_dim(
    X_train_device_id)

ffnn = FNN(X_train_prev_event_input_dim, X_train_prev_prev_event_input_dim, X_train_session_id_input_dim, X_train_device_id_input_dim, X_train,
           y_train, unique_classifications, X_train_prev_event, X_train_prev_prev_event, X_train_session_id, X_train_device_id, X_train_content_id)

study = ffnn.hypertune_model(n_trials=10, direction='maximize')

best_trial = study.best_trial

X_train_inputs = [X_train_prev_event, X_train_prev_prev_event,
                  X_train_session_id, X_train_device_id, X_train, X_train_content_id]
X_test_inputs = [X_test_prev_event, X_test_prev_prev_event,
                 X_test_session_id, X_test_device_id, X_test, X_test_content_id]

eval_data = []
for i in range(0, 5):
    eval_data.append(ffnn.train_model(best_trial, X_train_inputs,
                     X_test_inputs, y_test, 200, 32))

ffnn.visualizeTrainingResults(eval_data)
