import datetime
from dotenv import load_dotenv
import json
import numpy as np
import os
import pandas as pd
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
import time
from tqdm import tqdm
from typing import Dict, List, Tuple


def format_movie_id_col_and_update_dtypes(data: pd.DataFrame) -> pd.DataFrame:
    """This helper function extracts the movie IDs from the column "user_id" and saves them into a separate column. Also it turns the "_id" columns into type int and the "date" column into pandas date format."""
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
    )[["movie_id", "year"]]
    data = format_movie_id_col_and_update_dtypes(data)
    data = data.merge(movie_data, on="movie_id")
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
    return_dict = {}
    for col in cols:
        return_dict[col] = np.array(pd.DataFrame(data)[col].values)
    return return_dict


def split_data_into_x_train_etc(data: pd.DataFrame, y_col: str,
                                x_cols: List[str]) -> Dict[str, np.array]:
    """Extracts training, validation and test data from the main dataframe. All x-variables are normalized between -1 and 1.
    """
    train_data, temp_test_data = train_test_split(data,
                                                  test_size=0.3,
                                                  random_state=42)
    test_data, valid_data = train_test_split(temp_test_data,
                                             test_size=0.5,
                                             random_state=42)
    return {
        "x_train":
        turn_pandas_df_into_dict_of_np_arrays_of_selected_columns(
            train_data, x_cols),
        "x_valid":
        turn_pandas_df_into_dict_of_np_arrays_of_selected_columns(
            valid_data, x_cols),
        "x_test":
        turn_pandas_df_into_dict_of_np_arrays_of_selected_columns(
            test_data, x_cols),
        "y_train":
        train_data[y_col],
        "y_valid":
        valid_data[y_col],
        "y_test":
        test_data[y_col]
    }


def create_optimizer_w_learning_rate(
        opt_name: str, learning_rate: float) -> tf.keras.optimizers:
    if opt_name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)
    if opt_name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    return "error"


def load_json_param() -> Dict:
    return json.load(open("./model_parameters.json"))


def get_hparam(parameter_name: str) -> hp.HParam:
    """Accesses model_parameters.json by arg parameter_name as dict key and returns respective value as tensorboard.plugins.hparams object."""
    return hp.HParam(parameter_name,
                     hp.Discrete(load_json_param()[parameter_name]))


def get_all_hparams() -> Tuple[hp.HParam]:
    """Uses get_hparam() to get all parameters that are tuned in tune_model().
    """
    return get_hparam('num_units'), get_hparam('num_layers'), get_hparam(
        'optimizer'), get_hparam('learning_rate'), get_hparam('batch_size')


def get_log_name_with_current_timestamp() -> str:
    datetime_now = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    return f'logs_{datetime_now}/hparam_tuning'


def log_session(session_name: str) -> None:
    """Creates log folder in root directory and sets up logfile structure for hparams view in tensorboard.
    """
    with tf.summary.create_file_writer(session_name).as_default():
        hp.hparams_config(
            hparams=[*get_all_hparams()],
            metrics=[
                hp.Metric(load_json_param()['metric_accuracy'],
                          display_name='Accuracy')
            ],
        )


def print_tensorboard_bash_command(log_name: str) -> None:
    """Prints bash command for opening log file of current tuning run in browser. Highly recommended if many tuning parameters are run at once.
    """
    print("tensorboard --logdir " + log_name[:20] + " --port " +
          log_name[14:18])


def helper_fct_return_optimizer_w_learn_rate(opt_name: str, lr: float):
    if opt_name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr)
    if opt_name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    return "error"


def build_model(hparams: Dict[str, int], raw_data: pd.DataFrame) -> float:
    body = tf.keras.models.Sequential()
    for _ in range(int(hparams["HP_NUM_LAYERS"])):
        body.add(tf.keras.layers.Dense(hparams["HP_NUM_UNITS"]))
    body.add(tf.keras.layers.Dense(5))
    input_tensors = create_input_tensors(raw_data)
    preprocessing_head = create_preprocessing_for_model(
        raw_data, input_tensors)
    preprocessed_inputs = preprocessing_head(input_tensors)
    result = body(preprocessed_inputs)
    model = tf.keras.Model(input_tensors, result)
    model.compile(
        optimizer=helper_fct_return_optimizer_w_learn_rate(
            hparams["HP_OPTIMIZER"],
            hparams["HP_LEARNING_RATE"],
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[load_json_param()["metric_accuracy"]],
    )
    return model


def train_model(hparams: Dict[str, int], run_dir: str,
                tuning_input_data: Dict[str, np.array], model) -> float:
    model.fit(
        tuning_input_data["x_train"],
        tuning_input_data["y_train"],
        validation_data=(tuning_input_data["x_valid"],
                         tuning_input_data["y_valid"]),
        epochs=load_json_param()["epochs"],
        shuffle=True,
        verbose=True,
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=run_dir + '_' + str(hparams["HP_NUM_LAYERS"]) +
                'layers_' + str(hparams["HP_NUM_UNITS"]) + 'nodes_' +
                hparams["HP_OPTIMIZER"] + str(hparams["HP_LEARNING_RATE"]) +
                '_' + str(hparams["HP_BATCH_SIZE"]),
                histogram_freq=1)
        ],
        batch_size=(hparams["HP_BATCH_SIZE"]))
    return model


def tune_model(log_name: str, tuning_input_data: Dict[str, np.array],
               raw_data: pd.DataFrame) -> None:
    """"Loops through hparam combinations. For each combination it trains the ANN and logs its metrics in two ways: Training and evaluation is logged in train_model(). Testing is logged separately afterwards. Both can be seen in tensorboard.
    """
    session_num = 0
    for hparams_combination in tqdm(product(
            get_hparam('num_units').domain.values,
            get_hparam('num_layers').domain.values,
            get_hparam('optimizer').domain.values,
            get_hparam('learning_rate').domain.values,
            get_hparam('batch_size').domain.values),
                                    desc="Tuning hyper parameters"):
        hparams = dict(
            zip(("HP_NUM_UNITS", "HP_NUM_LAYERS", "HP_OPTIMIZER",
                 "HP_LEARNING_RATE", "HP_BATCH_SIZE"), hparams_combination))
        run_dir = f'{log_name}_run-{session_num}'
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(dict(zip((get_all_hparams()), hparams_combination)))
            model = train_model(hparams, run_dir, tuning_input_data,
                                build_model(hparams, raw_data))
            tf.summary.scalar(load_json_param()["metric_accuracy"],
                              model.evaluate(tuning_input_data["x_test"],
                                             tuning_input_data["y_test"],
                                             verbose=False)[1],
                              step=1)
        session_num += 1


def create_input_tensors(data: pd.DataFrame) -> Dict:
    inputs = {}
    for name, column in data.drop('rating', axis=1).items():
        dtype = column.dtype
        if dtype == object:
            dtype = tf.string
        else:
            dtype = tf.float32
        inputs[name] = tf.keras.Input(shape=(1, ), name=name, dtype=dtype)
    return inputs


def preprocess_and_concatenate_tensors(data: pd.DataFrame, input_tensors: Dict):
    """to do...
    """    
    numeric_inputs = {
        name: input
        for name, input in input_tensors.items() if input.dtype == tf.float32
    }
    tensor_numeric_inputs = tf.keras.layers.Concatenate()(list(
        numeric_inputs.values()))
    norm = tf.keras.layers.Normalization()
    norm.adapt(np.array(data[numeric_inputs.keys()]))
    all_numeric_inputs = norm(tensor_numeric_inputs)
    preprocessed_inputs = [all_numeric_inputs]
    for name, tensor in input_tensors.items():
        if tensor.dtype == tf.float32:
            continue
        lookup = tf.keras.layers.StringLookup(
            vocabulary=np.unique(data.drop('rating', axis=1)[name]))
        one_hot = tf.keras.layers.CategoryEncoding(
            num_tokens=lookup.vocabulary_size())
        preprocessed_inputs.append(one_hot(lookup(tensor)))
    return  tf.keras.layers.Concatenate()(preprocessed_inputs)

# def create_preprocessing_for_model(data:pd.DataFrame) ->Tuple[tf.keras.engine.functional.Functional,Dict[str,tf.keras.engine.keras_tensor.KerasTensor]]: #tpye hint issue here!
def create_preprocessing_for_model(data: pd.DataFrame, input_tensors: Dict):
    """Creates preprocessing object which is...?
    """
    preprocessed_inputs_concatenated = preprocess_and_concatenate_tensors(data,input_tensors)
    data_preprocessing = tf.keras.Model(input_tensors, preprocessed_inputs_concatenated)
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
    rating_data = pd.read_pickle(
        "C:/Users/reifv/root/Heidelberg Master/Netflix_AI_codes/data_densed_7557rows.pkl"
    )
    show_dataframe(rating_data)

    rating_data[
        'rating'] -= 1  #these 3 lines will be merged into parse...() fct or somewhere else
    rating_data = rating_data.astype({'movie_id': 'str', 'user_id': 'str'})
    rating_data.pop("date")

    model_input_data = split_data_into_x_train_etc(
        rating_data, y_col="rating", x_cols=["user_id", "movie_id", "year"])
    log_name = 'logs_' + datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S") + '/hparam_tuning'
    log_session(log_name)
    tune_model(log_name, model_input_data, rating_data)


if __name__ == "__main__":
    main()
# preprocessing noch weiter refactoren! vllt numeric und non numeric inputs noch trennen... immer an single resp denken...
# nach prepro falls nichts anderes mehr datenimport auf normal und dann pr!