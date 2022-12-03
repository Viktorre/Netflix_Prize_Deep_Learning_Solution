import pandas as pd
import pandas.api.types as ptypes
from unittest import TestCase
from hypothesis import given, settings, strategies as st
from hypothesis.extra.pandas import column, data_frames
from main import format_movie_id_col_and_update_dtypes


class TestFormatMainData(TestCase):

    expected_column_names = ['user_id', 'rating', 'date', 'movie_id']

    # @given(st.datetimes())
    @given(data_frames([column('user_id', dtype=str), column('rating', dtype=int),column('date', dtype="datetime64[ns]")]))
    # @settings(deadline=20000)
    def test_example_test_method(self, dataframe):
        self.assertTrue(True)

    def test_column_names(self):
        rating_test_data = self.__create_test_dataframe()
        rating_test_data = format_movie_id_col_and_update_dtypes(
            rating_test_data)
        self.assertListEqual(rating_test_data.columns.tolist(),
                             self.expected_column_names)

    def test_dtypes_conversion(self):
        rating_test_data = self.__create_test_dataframe()
        rating_test_data = format_movie_id_col_and_update_dtypes(
            rating_test_data)
        self.assertTrue(
            ptypes.is_datetime64_any_dtype(rating_test_data['date']))
        numeric_columns = ['user_id', 'rating', 'movie_id']
        for numeric_column in numeric_columns:
            self.assertTrue(
                ptypes.is_numeric_dtype(rating_test_data[numeric_column]))

    def __create_test_dataframe(self) -> pd.DataFrame:
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