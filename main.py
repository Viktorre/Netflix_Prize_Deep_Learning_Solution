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


def scale_and_split_data_into_x_train_etc(
        data: pd.DataFrame, y_col: str,
        x_cols: List[str]) -> Dict[str, pd.DataFrame]:
    """Extracts training, validation and test data from the main dataframe. All x-variables are normalized between -1 and 1.
    """
    train_data, temp_test_data = train_test_split(data,
                                                  test_size=0.3,
                                                  random_state=42)
    test_data, valid_data = train_test_split(temp_test_data,
                                             test_size=0.5,
                                             random_state=42)
    return {
        "y_train": get_y(train_data, y_col),
        "x_train": get_and_scale_x(train_data, x_cols),
        "y_valid": get_y(valid_data, y_col),
        "x_valid": get_and_scale_x(valid_data, x_cols),
        "y_test": get_y(test_data, y_col),
        "x_test": get_and_scale_x(test_data, x_cols)
    }


def create_optimizer_w_learning_rate(opt_name: str,
                                     learning_rate: float) -> tf.keras.optimizers:
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
       
def get_all_hparams()->Tuple[hp.HParam]:
    """Uses get_hparam() to get all parameters that are tuned in tune_model().
    """    
    return get_hparam('num_units'),get_hparam('num_layers'),get_hparam('optimizer'),get_hparam('learning_rate'),get_hparam('batch_size')

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

def helper_fct_return_optimizer_w_learn_rate(opt_name:str,lr:float):
    if opt_name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr)
    if opt_name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    return "error"
    
def train_test_model(hparams,run_dir,preprocessing_head, inputs,HP_NUM_UNITS,HP_NUM_LAYERS,HP_OPTIMIZER,HP_LEARNING_RATE,HP_BATCH_SIZE,METRIC_ACCURACY,x_train,y_train,x_valid,y_valid,x_test, y_test):
    body = tf.keras.models.Sequential()
    for _ in range(int(hparams[HP_NUM_LAYERS])):
        body.add(tf.keras.layers.Dense(hparams[HP_NUM_UNITS]))
    body.add(tf.keras.layers.Dense(5))
    preprocessed_inputs = preprocessing_head(inputs)
    result = body(preprocessed_inputs)
    model = tf.keras.Model(inputs, result)
    model.compile(
        optimizer=helper_fct_return_optimizer_w_learn_rate(hparams[HP_OPTIMIZER],hparams[HP_LEARNING_RATE],),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[METRIC_ACCURACY],)
    model.fit(x_train, y_train,validation_data=(x_valid,y_valid),epochs=load_json_param()["epochs"], shuffle=True,verbose=True, callbacks=[ tf.keras.callbacks.TensorBoard(log_dir=run_dir+'_'+str(hparams[HP_NUM_LAYERS])+'layers_'+str(hparams[HP_NUM_UNITS])+'nodes_'+hparams[HP_OPTIMIZER]+str(hparams[HP_LEARNING_RATE])+'_'+str(hparams[HP_BATCH_SIZE]), histogram_freq=1)],batch_size=(hparams[HP_BATCH_SIZE])) 
    _, accuracy = model.evaluate(x_test, y_test,verbose=False)
    return accuracy


def run(run_dir, hparams, preprocessing_head, inputs, HP_NUM_UNITS,HP_NUM_LAYERS,HP_OPTIMIZER,HP_LEARNING_RATE,HP_BATCH_SIZE,METRIC_ACCURACY,x_train,y_train,x_valid,y_valid,x_test, y_test):
    hparams_dict_for_logging = hparams
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams_dict_for_logging)  # design bottle neck: this fct needs a dict that has all hparams as hp object as keys and the respective parameter value from the loop as value.
        accuracy = train_test_model(hparams,run_dir, preprocessing_head, inputs,HP_NUM_UNITS,HP_NUM_LAYERS,HP_OPTIMIZER,HP_LEARNING_RATE,HP_BATCH_SIZE,METRIC_ACCURACY,x_train,y_train,x_valid,y_valid,x_test, y_test)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
        
def tune_model(log_name,rating_data_preprocessing, inputs, HP_NUM_UNITS,HP_NUM_LAYERS,HP_OPTIMIZER,HP_LEARNING_RATE,HP_BATCH_SIZE,METRIC_ACCURACY,x_train,y_train,x_valid,y_valid,x_test, y_test) ->None:
    session_num = 0
    for hparams_combination in tqdm(product(
            HP_NUM_UNITS.domain.values,
            HP_NUM_LAYERS.domain.values,
            HP_OPTIMIZER.domain.values,
            HP_LEARNING_RATE.domain.values,
            HP_BATCH_SIZE.domain.values),desc="Tuning hyper parameters"):
        run_name = f'run-{session_num}'
        hparams = dict( zip((HP_NUM_UNITS, HP_NUM_LAYERS, HP_OPTIMIZER, HP_LEARNING_RATE, HP_BATCH_SIZE), hparams_combination))
        # hparams = dict( zip(get_all_hparams(), hparams_combination))
        run( f'{log_name}_{run_name}', hparams,rating_data_preprocessing, inputs, HP_NUM_UNITS,HP_NUM_LAYERS,HP_OPTIMIZER,HP_LEARNING_RATE,HP_BATCH_SIZE,METRIC_ACCURACY,x_train,y_train,x_valid,y_valid,x_test, y_test)
        session_num += 1
        
    
def main() -> None:
    print("fun todo: refactor this, only then go back to normal data import!")
    load_dotenv('.env.md')
    rating_data = pd.read_pickle("C:/Users/reifv/root/Heidelberg Master/Netflix_AI_codes/data_densed_7557rows.pkl")
    show_dataframe(rating_data)

    rating_data['rating'] -=1
    rating_data = rating_data.astype({ 'movie_id': 'str', 'user_id': 'str' })
    rating_data.pop("date")
    rating_data_features = rating_data.copy()
    rating_data_labels = rating_data_features.pop('rating')
    inputs = {}

    for name, column in rating_data_features.items():
        dtype = column.dtype
        if dtype == object:
            dtype = tf.string
        else:
            dtype = tf.float32
        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)


    numeric_inputs = {name:input for name,input in inputs.items()
                    if input.dtype==tf.float32}

    x = tf.keras.layers.Concatenate()(list(numeric_inputs.values()))
    norm = tf.keras.layers.Normalization()
    norm.adapt(np.array(rating_data[numeric_inputs.keys()]))
    all_numeric_inputs = norm(x)
    preprocessed_inputs = [all_numeric_inputs]

    for name, input in inputs.items():
        if input.dtype == tf.float32:
            continue
        lookup = tf.keras.layers.StringLookup(vocabulary=np.unique(rating_data_features[name]))
        one_hot = tf.keras.layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())
        x = lookup(input)
        x = one_hot(x)
        preprocessed_inputs.append(x)

    preprocessed_inputs_cat = tf.keras.layers.Concatenate()(preprocessed_inputs)

    rating_data_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

    rating_data_features_dict = {name: np.array(value) 
                            for name, value in rating_data_features.items()}
    features_dict = {name:values[:1] for name, values in rating_data_features_dict.items()}

    rating_data_preprocessing(features_dict)
    
    x_train, x_valid, x_test = {}, {}, {}
    for col in pd.DataFrame(rating_data_features_dict):
        x_train[col] = np.array(pd.DataFrame(rating_data_features_dict)[col][:5000].values)
        x_valid[col] = np.array(pd.DataFrame(rating_data_features_dict)[col][5000:6200].values)
        x_test[col] = np.array(pd.DataFrame(rating_data_features_dict)[col][6200:].values)
    y_train, y_valid, y_test = rating_data_labels[:5000],rating_data_labels[5000:6200],   rating_data_labels[6200:]
    
    log_name = 'logs_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'/hparam_tuning'

    log_session(log_name)
    tune_model(log_name,rating_data_preprocessing, inputs, *get_all_hparams(),load_json_param()['metric_accuracy'],x_train,y_train,x_valid,y_valid,x_test, y_test)
    
    
if __name__ == "__main__":
    main()
