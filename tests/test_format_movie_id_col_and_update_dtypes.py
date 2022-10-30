import pandas as pd
from unittest import TestCase
from main import format_movie_id_col_and_update_dtypes


class TestFormatMainData(TestCase):

    expected_column_names = ['user_id', 'rating', 'date', 'movie_id']

    # expected_dtypes = [dtype('int64'), dtype('int64'), dtype('<M8[ns]'), dtype('int64')]

    def test_dtypes_conversion_and_column_names(self):
        df = pd.DataFrame({
            'user_id': ['1:', '212101', '312101', '2:', '401211', '501211'],
            'rating': [pd.NA, 2, 3, pd.NA, 5, 5],
            'date': [
                pd.NA,
                pd.to_datetime('20101231'),
                pd.to_datetime('20101231'), pd.NA,
                pd.to_datetime('20101231'),
                pd.to_datetime('20101231')
            ]
        })
        df = format_movie_id_col_and_update_dtypes(df)

        self.assertListEqual(df.columns.tolist(), self.expected_column_names)
