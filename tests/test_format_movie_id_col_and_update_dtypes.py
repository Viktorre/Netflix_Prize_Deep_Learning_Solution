import pandas as pd
import pandas.api.types as ptypes
from unittest import TestCase
from main import format_movie_id_col_and_update_dtypes


class TestFormatMainData(TestCase):

    expected_column_names = ['user_id', 'rating', 'date', 'movie_id']

    def create_test_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
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

    def test_column_names(self):
        df = self.create_test_dataframe()
        df = format_movie_id_col_and_update_dtypes(df)
        self.assertListEqual(df.columns.tolist(), self.expected_column_names)

    def test_dtypes_conversion(self):
        df = self.create_test_dataframe()
        df = format_movie_id_col_and_update_dtypes(df)
        self.assertTrue(ptypes.is_datetime64_any_dtype(df['date']))
        for column_that_should_be_numeric in ['user_id', 'rating', 'movie_id']:
            self.assertTrue(
                ptypes.is_numeric_dtype(df[column_that_should_be_numeric]))
