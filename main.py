import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import datetime
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from tqdm import tqdm


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


def parse_format_and_join_data(path) -> pd.DataFrame:
    df = pd.read_csv(path + "combined_data_1_short.txt")
    df = format_df(df)
    movie_df = pd.read_csv(path + "movie_titles.csv",
                           header=0)[["movie_id", "year"]]
    df = df.merge(movie_df, on="movie_id")
    return df


def main() -> None:
    path = "C:/Users/reifv/root/Heidelberg Master/vs_codes/netflix_project/"
    df = parse_format_and_join_data(path)
    print(df.head())
    df = df[]
    train_dataset, temp_test_dataset =  train_test_split(df, test_size=0.3,random_state=42)
    test_dataset, valid_dataset =  train_test_split(temp_test_dataset, test_size=0.5,random_state=42)
    train_labels = train_dataset.pop('rating')
    test_labels = test_dataset.pop('rating')
    valid_labels = valid_dataset.pop('rating')
    train_stats = train_dataset.describe()
    train_stats = train_stats.transpose()
    normed_train_data = pd.DataFrame(StandardScaler().fit_transform(train_dataset), columns=train_dataset.columns, index=train_dataset.index)
    normed_test_data = pd.DataFrame(StandardScaler().fit_transform(test_dataset), columns=test_dataset.columns, index=test_dataset.index)
    normed_valid_data = pd.DataFrame(StandardScaler().fit_transform(valid_dataset), columns=valid_dataset.columns, index=valid_dataset.index)
    x_train, y_train, x_valid, y_valid = normed_train_data, train_labels, normed_valid_data, valid_labels



    print(1)

if __name__ == "__main__":
    main()
