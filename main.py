import json
from typing import Union
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import datetime
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from tqdm import tqdm


def format_data(data: pd.DataFrame) -> pd.DataFrame:
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    mask = np.logical_and(data['rating'].isnull(), data['date'].isnull())
    data['movie_id'] = np.nan
    data['movie_id'] = data.loc[mask, 'user_id'].str.extract('(\d+)')
    data['movie_id'] = data['movie_id'].ffill()
    data = data.loc[~mask]
    data = data.astype({'movie_id': 'int', 'user_id': 'int'})
    return data


def prepare_drive_link(url: str) -> str:
    return 'https://drive.google.com/uc?export=download&id=' + url.split(
        '/')[-2]


def parse_format_and_join_data(url_main_file: str,
                               url_movie_file: str) -> pd.DataFrame:
    data = pd.read_csv(prepare_drive_link(url_main_file),
                       sep=',',
                       na_values=[''],
                       names=['user_id', 'rating', 'date'],
                       dtype={
                           'user_id': 'string',
                           'rating': 'Int64',
                           'date': 'string'
                       })
    movie_data = pd.read_csv(
        prepare_drive_link(url_movie_file),
        header=0,
    )[["movie_id", "year"]]
    data = format_data(data)
    data = data.merge(movie_data, on="movie_id")
    return data


def show_dataframe(data: pd.DataFrame) -> None:
    print(data.info())
    print(data.head())


def main() -> None:
    parameters = json.load(open(file="./parameters.json", encoding="utf-8"))
    rating_data = parse_format_and_join_data(
        url_main_file=parameters["url_short_main_file"],
        url_movie_file=parameters["url_movie_info_file"],
    )
    show_dataframe(rating_data)


if __name__ == "__main__":
    main()
