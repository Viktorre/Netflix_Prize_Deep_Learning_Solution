import json
import os
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
from dotenv import load_dotenv


def format_movie_id_col_and_update_dtypes(data: pd.DataFrame) -> pd.DataFrame:
    """ Reformats movie_id, which before were in seperate lines, into a new column and deletes obsolete lines. Returns data containing each information as a separate column with adequate data types. 
    """
    mask = np.logical_and(data['rating'].isnull(), data['date'].isnull())
    data['movie_id'] = np.nan
    data['movie_id'] = data.loc[mask, 'user_id'].str.extract('(\d+)')
    data['movie_id'] = data['movie_id'].ffill()
    data = data.loc[~mask]
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data = data.astype({'movie_id': 'int64', 'user_id': 'int64'})
    return data


def prepare_drive_link(url: str) -> str:
    return 'https://drive.google.com/uc?export=download&id=' + url.split(
        '/')[-2]


def parse_format_and_join_data() -> pd.DataFrame:
    """ Parses the main movie rating file and movie info file. Formats the main movie rating file and joins it with the movie info file.
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


def main() -> None:
    load_dotenv('.env.md')
    rating_data = parse_format_and_join_data()
    show_dataframe(rating_data)


if __name__ == "__main__":
    main()
