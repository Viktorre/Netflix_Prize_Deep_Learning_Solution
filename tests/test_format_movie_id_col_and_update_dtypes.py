from unittest import TestCase
from main import format_movie_id_col_and_update_dtypes

class TestFormatMainData(TestCase):
    
    def test_dtypes_conversion_and_column_names(self):
        # df = pd.DataFrame{[asdfkljasdlkfj,aglkasdfjasdklfjaslkdfjasdf]}
        # df = format_movie_id_col_and_update_dtypes(df)
        print(format_movie_id_col_and_update_dtypes)
        
        self.assertEqual([1],[1])
    