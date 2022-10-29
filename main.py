# -*- coding: utf-8 -*-
import datetime
import json
import os
from itertools import product
from typing import Dict, List, Tuple

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorboard.plugins.hparams import api as hp
from tqdm import tqdm


def format_movie_id_col_and_update_dtypes(data: pd.DataFrame) -> pd.DataFrame:
    """This helper function extracts the movie IDs from the column "user_id"
    and saves them into a separate column.

    Also it turns the "_id" columns into type int and the "date" column
    into pandas date format.
    """
    mask = np.logical_and(data['rating'].isnull(), data['date'].isnull())
    data['movie_id'] = np.nan
    data['movie_id'] = data.loc[mask, 'user_id'].str.extract('(\d+)')
    data['movie_id'] = data['movie_id'].ffill()
    data = data.loc[~mask]
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data = data.astype({
        'movie_id': 'int64',
        'user_id': 'int64',
        'rating': 'int64'
    })
    return data


def subtract_rating_integers_by_one(data: pd.DataFrame) -> pd.DataFrame:
    """Turns rating integers from [1,2,3,4,5] to [0,1,2,3,4].

    This is needed for the tensor model.
    """
    data['rating'] -= 1
    return data


def prepare_drive_link(url: str) -> str:
    return 'https://drive.google.com/uc?export=download&id=' + url.split(
        '/')[-2]


def parse_and_join_data() -> pd.DataFrame:
    """The 2 following movie files are being imported, parsed and joined:

    - ratings
    - (additional) info
    """
    data = pd.read_csv(prepare_drive_link(os.getenv('url_short_main_file')),
                       sep=',',
                       na_values=[''],
                       names=['user_id', 'rating', 'date'],
                       dtype={
                           'user_id': 'string',
                           'rating': 'Int64',
                           'date': 'string'
                       })
    movie_data = pd.read_csv(
        prepare_drive_link(os.getenv('url_movie_info_file')),
        header=0,
    )[['movie_id', 'year']]
    data = format_movie_id_col_and_update_dtypes(data)
    data = subtract_rating_integers_by_one(data)
    data = data.merge(movie_data, on='movie_id')
    data = data.astype({'movie_id': 'str', 'user_id': 'str'})
    data.pop('date')
    return data


def show_dataframe(data: pd.DataFrame) -> None:
    print(data.info())
    print(data.head())


def get_y(data: pd.DataFrame, y_col: str) -> pd.DataFrame:
    return data[y_col]


def get_and_scale_x(data: pd.DataFrame, x_cols: List[str]) -> pd.DataFrame:
    data = data[x_cols]
    return pd.DataFrame(StandardScaler().fit_transform(data),
                        columns=data.columns,
                        index=data.index)


def turn_pandas_df_into_dict_of_np_arrays_of_selected_columns(
        data: pd.DataFrame, cols: List[str]) -> Dict[str, np.array]:
    """For each column of all select columns in "cols" list, this function
    stores the values of that column as a numpy array in dictionary with the
    column name as key.

    Dtypes from the input dataframe remain unchanged in the dictionary.
    """
    return {col: data[col].values for col in cols}


def split_data_into_x_train_etc(data: pd.DataFrame, y_col: str,
                                x_cols: List[str]) -> Dict[str, np.array]:
    """Extracts training, validation and test data from the main dataframe.

    All x-variables are normalized between -1 and 1.
    """
    train_data, temp_test_data = train_test_split(data,
                                                  test_size=0.3,
                                                  random_state=42)
    test_data, valid_data = train_test_split(temp_test_data,
                                             test_size=0.5,
                                             random_state=42)
    return {
        'x_train':
        turn_pandas_df_into_dict_of_np_arrays_of_selected_columns(
            train_data, x_cols),
        'x_valid':
        turn_pandas_df_into_dict_of_np_arrays_of_selected_columns(
            valid_data, x_cols),
        'x_test':
        turn_pandas_df_into_dict_of_np_arrays_of_selected_columns(
            test_data, x_cols),
        'y_train':
        train_data[y_col],
        'y_valid':
        valid_data[y_col],
        'y_test':
        test_data[y_col]
    }


def load_json_params() -> Dict:
    return json.load(open('./model_parameters.json'))


def get_hparam(parameter_name: str) -> hp.HParam:
    """Accesses model_parameters.json by arg parameter_name as dict key and
    returns respective value as tensorboard.plugins.hparams object."""
    return hp.HParam(parameter_name,
                     hp.Discrete(load_json_params()[parameter_name]))


def get_all_hparams() -> Tuple[hp.HParam]:
    """Uses get_hparam() to get all parameters that are tuned in tune_model()
    and returns them as tuple in a specific order which is hard coded."""
    ordered_lists_of_hparams = (get_hparam('num_units'),
                                get_hparam('num_layers'),
                                get_hparam('optimizer'),
                                get_hparam('learning_rate'),
                                get_hparam('batch_size'))
    return ordered_lists_of_hparams


def get_log_name_with_current_timestamp() -> str:
    datetime_now = datetime.datetime.now().strftime('%Y%m%d-%H%M')
    return f'logs_{datetime_now}/hparam_tuning'


def log_session(session_name: str) -> None:
    """Creates log folder in root directory and sets up logfile structure for
    hparams view in tensorboard."""
    with tf.summary.create_file_writer(session_name).as_default():
        hp.hparams_config(
            hparams=[*get_all_hparams()],
            metrics=[
                hp.Metric(load_json_params()['metric_accuracy'],
                          display_name='Accuracy')
            ],
        )


def print_tensorboard_bash_command(log_name: str) -> None:
    """Prints bash command for opening log file of current tuning run in
    browser.

    Highly recommended if many tuning parameters are run at once.
    """
    print('tensorboard --logdir ' + log_name[:20] + ' --port ' +
          log_name[14:18])


def helper_fct_return_optimizer_w_learn_rate(opt_name: str,
                                             lr: float) -> tf.keras.optimizers:
    if opt_name == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=lr)
    if opt_name == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=lr)
    raise ValueError(
        'optimizer_name not recognized or implemented in this fct')


def build_model(hparams: Dict[str, int],
                raw_data: pd.DataFrame) -> keras.engine.functional.Functional:
    body = tf.keras.models.Sequential()
    for _ in range(int(hparams['HP_NUM_LAYERS'])):
        body.add(tf.keras.layers.Dense(hparams['HP_NUM_UNITS']))
    body.add(tf.keras.layers.Dense(5))
    input_tensors = create_input_tensors(raw_data)
    preprocessing_head = create_preprocessing_for_model(
        raw_data, input_tensors)
    preprocessed_inputs = preprocessing_head(input_tensors)
    result = body(preprocessed_inputs)
    model = tf.keras.Model(input_tensors, result)
    model.compile(
        optimizer=helper_fct_return_optimizer_w_learn_rate(
            hparams['HP_OPTIMIZER'],
            hparams['HP_LEARNING_RATE'],
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[load_json_params()['metric_accuracy']],
    )
    return model


def train_model(hparams: Dict[str, int], run_dir: str,
                tuning_input_data: Dict[str, np.array],
                model) -> keras.engine.functional.Functional:
    model.fit(
        tuning_input_data['x_train'],
        tuning_input_data['y_train'],
        validation_data=(tuning_input_data['x_valid'],
                         tuning_input_data['y_valid']),
        epochs=load_json_params()['epochs'],
        shuffle=True,
        verbose=True,
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=run_dir + '_' + str(hparams['HP_NUM_LAYERS']) +
                'layers_' + str(hparams['HP_NUM_UNITS']) + 'nodes_' +
                hparams['HP_OPTIMIZER'] + str(hparams['HP_LEARNING_RATE']) +
                '_' + str(hparams['HP_BATCH_SIZE']),
                histogram_freq=1)
        ],
        batch_size=(hparams['HP_BATCH_SIZE']))
    return model


def tune_model(log_name: str, tuning_input_data: Dict[str, np.array],
               raw_data: pd.DataFrame) -> None:
    """"Loops through hparam combinations.

    For each combination it trains the ANN and logs its metrics in two
    ways: Training and evaluation is logged in train_model(). Testing is
    logged separately afterwards. Both can be seen in tensorboard.
    """
    session_num = 0
    for hparams_combination in tqdm(product(
            get_hparam('num_units').domain.values,
            get_hparam('num_layers').domain.values,
            get_hparam('optimizer').domain.values,
            get_hparam('learning_rate').domain.values,
            get_hparam('batch_size').domain.values),
                                    desc='Tuning hyper parameters'):
        hparams = dict(
            zip(('HP_NUM_UNITS', 'HP_NUM_LAYERS', 'HP_OPTIMIZER',
                 'HP_LEARNING_RATE', 'HP_BATCH_SIZE'), hparams_combination))
        run_dir = f'{log_name}_run-{session_num}'
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(dict(zip((get_all_hparams()), hparams_combination)))
            model = train_model(hparams, run_dir, tuning_input_data,
                                build_model(hparams, raw_data))
            tf.summary.scalar(load_json_params()['metric_accuracy'],
                              model.evaluate(tuning_input_data['x_test'],
                                             tuning_input_data['y_test'],
                                             verbose=False)[1],
                              step=1)
        session_num += 1


def create_input_tensors(
        data: pd.DataFrame
) -> Dict[str, keras.engine.keras_tensor.KerasTensor]:
    """Turns each dataframe column into a keras tensor and returns them as a
    dict."""
    tensors = {}
    for name, column in data.drop('rating', axis=1).items():
        dtype = column.dtype
        if dtype == object:
            dtype = tf.string
        else:
            dtype = tf.float32
        tensors[name] = tf.keras.Input(shape=(1, ), name=name, dtype=dtype)
    return tensors


def select_numeric_tensors(
    input_tensors: Dict[str, keras.engine.keras_tensor.KerasTensor]
) -> Dict[str, keras.engine.keras_tensor.KerasTensor]:
    """Returns all numeric tensors form input_tensors dictionary."""
    return {
        name: input
        for name, input in input_tensors.items() if input.dtype == float
    }


def select_non_numeric_tensors(
    input_tensors: Dict[str, keras.engine.keras_tensor.KerasTensor]
) -> Dict[str, keras.engine.keras_tensor.KerasTensor]:
    """Returns all non_numeric tensors form input_tensors dictionary."""
    return {
        name: input
        for name, input in input_tensors.items() if input.dtype != float
    }


def preprocess_numeric_tensors(
    data: pd.DataFrame, tensors: Dict[str,
                                      keras.engine.keras_tensor.KerasTensor]
) -> keras.engine.keras_tensor.KerasTensor:
    """Norms all tensors (must be numeric) and concats them into one tensor,
    which is returned.

    This return tensor has a shape of (None,num_of_numeric_features) as
    all numeric features are a number. That way all numeric features are
    represented in only one tensor.
    """
    layer_numeric_tensors = tf.keras.layers.Concatenate()(list(
        tensors.values()))
    norm = tf.keras.layers.Normalization()
    norm.adapt(np.array(data[tensors.keys()]))
    numeric_tensor_normed = norm(layer_numeric_tensors)
    return numeric_tensor_normed


def preprocess_non_numeric_tensors(
    data: pd.DataFrame, tensors: Dict[str,
                                      keras.engine.keras_tensor.KerasTensor]
) -> List[keras.engine.keras_tensor.KerasTensor]:
    """Encodes all tensor (non-numeric) as categories and returns them as a
    list of tensors.

    Each return tensor has a shape of (None,num_of_categories).
    """
    non_numeric_inputs_cat_encoded = []
    for name, tensor in tensors:
        lookup = tf.keras.layers.StringLookup(
            vocabulary=np.unique(data.drop('rating', axis=1)[name]))
        one_hot = tf.keras.layers.CategoryEncoding(
            num_tokens=lookup.vocabulary_size())
        non_numeric_inputs_cat_encoded.append(one_hot(lookup(tensor)))
    return non_numeric_inputs_cat_encoded


def preprocess_and_concatenate_tensors(
    data: pd.DataFrame,
    input_tensors: Dict[str, keras.engine.keras_tensor.KerasTensor]
) -> keras.engine.keras_tensor.KerasTensor:
    """Norms all numeric tensors and category-encodes all non-numeric tensors.

    Returns all pre-processed tensors as a concatenated keras layer.
    """
    return tf.keras.layers.Concatenate()([
        preprocess_numeric_tensors(data,
                                   select_numeric_tensors(input_tensors)),
        *preprocess_non_numeric_tensors(
            data,
            select_non_numeric_tensors(input_tensors).items())
    ])


def create_preprocessing_for_model(
    data: pd.DataFrame,
    input_tensors: Dict[str, keras.engine.keras_tensor.KerasTensor]
) -> keras.engine.functional.Functional:
    """Creates preprocessing object."""
    preprocessed_inputs_concatenated = preprocess_and_concatenate_tensors(
        data, input_tensors)
    data_preprocessing = tf.keras.Model(input_tensors,
                                        preprocessed_inputs_concatenated)
    data_features_dict = {
        name: np.array(value)
        for name, value in data.drop('rating', axis=1).items()
    }
    features_dict = {
        name: values[:1]
        for name, values in data_features_dict.items()
    }
    data_preprocessing(features_dict)
    return data_preprocessing


def main() -> None:
    load_dotenv('.env.md')
    rating_data = parse_and_join_data()
    show_dataframe(rating_data)
    model_input_data = split_data_into_x_train_etc(
        rating_data, y_col='rating', x_cols=['user_id', 'movie_id', 'year'])
    log_name = 'logs_' + datetime.datetime.now().strftime(
        '%Y%m%d-%H%M%S') + '/hparam_tuning'
    log_session(log_name)
    tune_model(log_name, model_input_data, rating_data)


if __name__ == '__main__':
    main()
