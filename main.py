import json
import os
from typing import Union, Dict, List
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import datetime
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv


def format_movie_id_col_and_update_dtypes(data: pd.DataFrame) -> pd.DataFrame:
    """This helper function extracts the movie IDs from the column "user_id" and saves them into a separate column. Also it turns the "_id" columns into type int and the "date" column into pandas date format."""
    mask = np.logical_and(data['rating'].isnull(), data['date'].isnull())
    data['movie_id'] = np.nan
    data['movie_id'] = data.loc[mask, 'user_id'].str.extract('(\d+)')
    data['movie_id'] = data['movie_id'].ffill()
    data = data.loc[~mask]
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data = data.astype({'movie_id': 'int64', 'user_id': 'int64','rating':int})
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


def create_optimizer_w_learning_rate(opt_name:str,lr:float) ->tf.keras.optimizers: #might be wrong return type hint. for example "adam"-optimizer has type <class 'keras.optimizers.optimizer_v2.adam.Adam'>
    if opt_name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr)
    if opt_name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    return "error"

def access_params_json() ->Dict:
    return json.load(open("./model_parameters.json"))

def get_hparam(parameter_name: str) ->hp:
    """Accesses model_parameters.json by arg parameter_name as dict key and returns respective value as tensorboard.plugins.hparams object."""    
    return hp.HParam(parameter_name, hp.Discrete(access_params_json()[parameter_name]))


def log_this_session(session_name:str) ->None:
    with tf.summary.create_file_writer(session_name).as_default(): #this creates logfile with hparams 
        hp.hparams_config(
            hparams=[get_hparam('num_units'), get_hparam('num_layers'), get_hparam('optimizer'), get_hparam('lr'), get_hparam('batch_size')],
            metrics=[hp.Metric(access_params_json()['metric_accuracy'], display_name='Accuracy')],
        )
        
        
def train_test_model(hparams,run_dir,input_data):
    body = tf.keras.models.Sequential()
    for _ in range(int(hparams["HP_NUM_LAYERS"])):
        body.add(tf.keras.layers.Dense(hparams["HP_NUM_UNITS"]))
    body.add(tf.keras.layers.Dense(1))
    model = body
    model.compile(
        optimizer=create_optimizer_w_learning_rate(hparams["HP_OPTIMIZER"],hparams["HP_LR"],),
        loss='BinaryCrossentropy', metrics=['accuracy'],)
    model.fit(input_data["x_train"], input_data["y_train"],validation_data=(input_data["x_valid"],input_data["y_valid"]),epochs=1, shuffle=True,verbose=True, callbacks=[ tf.keras.callbacks.TensorBoard(log_dir=run_dir+'_'+str(hparams["HP_NUM_LAYERS"])+'layers_'+str(hparams["HP_NUM_UNITS"])+'nodes_'+hparams["HP_OPTIMIZER"]+str(hparams["HP_LR"])+'_'+str(hparams["HP_BATCH_SIZE"])+'batchsize_', histogram_freq=1)],batch_size=(hparams["HP_BATCH_SIZE"])) 
    _, accuracy = model.evaluate(input_data["x_valid"], input_data["y_valid"],verbose=False)
    return accuracy

def run(run_dir, hparams, input_data):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(hparams,run_dir,input_data)
        tf.summary.scalar(access_params_json()["metric_accuracy"], accuracy, step=1)
        
        
def main() -> None:
    load_dotenv('.env.md')
    rating_data = parse_and_join_data()
    show_dataframe(rating_data)
    model_input_data = scale_and_split_data_into_x_train_etc(rating_data,
                                                             y_col="rating",
                                                             x_cols=["year"])
        
        
    log_name = 'logs_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'/hparam_tuning'
    log_this_session(log_name)

 
    session_num = 0
    for num_units in tqdm(get_hparam('num_units').domain.values):
        for num_layers in get_hparam('num_layers').domain.values:
            for optimizer in get_hparam('optimizer').domain.values:
                for lr in get_hparam('lr').domain.values:   
                    for batch_size in get_hparam('batch_size').domain.values:
                        hparams = {"HP_NUM_UNITS": num_units,"HP_NUM_LAYERS": num_layers, "HP_OPTIMIZER": optimizer, "HP_LR": lr, "HP_BATCH_SIZE": batch_size }
                        run_name = "run-%d" % session_num
                        run(log_name+ run_name, hparams, model_input_data)
                        session_num += 1

    # tb in browser:
    print("tensorboard --logdir "+log_name[:20]+" --port "+log_name[14:18])
    
    
if __name__ == "__main__":
    main()


#commit
# changed int64 to int due to tf issue
