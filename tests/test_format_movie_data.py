import os
import pandas as pd
from main import format_movie_id_col_and_update_dtypes

def test_format_movie_data() ->None:
    """...
    """    
    data = pd.read_csv("tests/data/data_movie_id_format.txt",
                       sep=',',
                       na_values=[''],
                       names=['user_id', 'rating', 'date'],
                       dtype={
                           'user_id': 'string',
                           'rating': 'Int64',
                           'date': 'string'
                       })
    formatted_data = format_movie_id_col_and_update_dtypes(data)
    assert len(data)>0
    