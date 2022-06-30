from typing import Tuple
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import datetime
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from tqdm import tqdm
import requests
from io import StringIO


def format_movie_id(df) -> pd.DataFrame:
    df = df.reset_index()
    new_col = []
    temp = "1:"
    for row in df.itertuples():
        if ":" in row[1]:
            temp = row[1]
        new_col.append(temp[:-1])
    df["movie_id"] = new_col
    return df


def format_df(df) -> pd.DataFrame:
    df = format_movie_id(df)
    df.columns = ["customer_id", "rating", "date", "movie_id"]
    df["movie_id"] = df["movie_id"].astype(int)
    df["date"] = pd.to_datetime(df["date"])
    return df


def prepare_drive_link(url) -> str:
    return 'https://drive.google.com/uc?export=download&id=' + url.split(
        '/')[-2]


def parse_format_and_join_data(drive_links) -> pd.DataFrame:
    df = pd.read_csv(prepare_drive_link(drive_links["short_main_file"]))
    df = format_df(df)
    movie_df = pd.read_csv(prepare_drive_link(drive_links["movie_info_csv"]),
                           header=0)[["movie_id", "year"]]
    df = df.merge(movie_df, on="movie_id")
    return df


def get_y(data: pd.DataFrame, y_col: str) -> pd.DataFrame:
    return data[y_col]


def get_and_scale_x(data: pd.DataFrame, x_cols: list[str]) -> pd.DataFrame:
    data = data[x_cols]
    return pd.DataFrame(StandardScaler().fit_transform(data),
                        columns=data.columns,
                        index=data.index)


def scale_and_split_data_into_x_train_etc(
        data: pd.DataFrame, y_col: str,
        x_cols: list[str]) -> Tuple[pd.DataFrame]:
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


def main() -> None:
    drive_links = {
        "short_main_file":
        "https://drive.google.com/file/d/1HAy11Oa03iMbKVbNhIN8JAp576U-py_T/view?usp=sharing",
        "movie_info_csv":
        "https://drive.google.com/file/d/1--R03vOj24Tnc4hOJxdEKkqu5q_yIiH8/view?usp=sharing"
    }
    rating_data = parse_format_and_join_data(drive_links)
    print(rating_data.info())
    print(rating_data.head())

    rating_data = rating_data.dropna()
    rating_data = rating_data.astype({
        'movie_id': 'int64',
        'customer_id': 'int64',
        "rating": "int64"
    })
    model_input_data = scale_and_split_data_into_x_train_etc(rating_data,
                                                             y_col="rating",
                                                             x_cols=["year"])


if __name__ == "__main__":
    main()
