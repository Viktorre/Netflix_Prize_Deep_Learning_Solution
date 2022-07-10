import os
import pandas as pd
from main import format_movie_id_col_and_update_dtypes


def test_format_movie_data() -> None:
    """Uses short sample file to check the following points in format_movie_id_col_and_update_dtypes():
        1. Does the column "movie_id" exist?
        2. Is the "movie_id" correctly mapped to the "user_id"?
        3. Are the redundant rows containing the now mapped movie ids deleted?
    """
    data = pd.read_csv('tests/data/data_movie_id_format.txt',
                       sep=',',
                       na_values=[''],
                       names=['user_id', 'rating', 'date'],
                       dtype={
                           'user_id': 'string',
                           'rating': 'Int64',
                           'date': 'string'
                       })
    formatted_data = format_movie_id_col_and_update_dtypes(data)
    assert formatted_data.shape[1] == 4
    assert formatted_data.shape[0] == 12
    assert len(formatted_data["movie_id"].value_counts().unique()) == 1
