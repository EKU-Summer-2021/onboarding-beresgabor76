import unittest
import urllib
import pandas as pd
from src.file_reader import read_file


class TestFileReading(unittest.TestCase):

    def test_read_when_correct_url(self):
        url = "https://raw.githubusercontent.com/EKU-Summer-2021/intelligent_system_data/" + \
              "main/Intelligent%20System%20Data/CSP/CSP_360.csv"
        self.assertEqual(True, isinstance(read_file(url), pd.DataFrame))

    def test_read_when_incorrect_url(self):
        url = "https://r.githubusercontent.com/EKU-Summer-2021/intelligent_system_data/" + \
              "main/Intelligent%20System%20Data/CSP/CSP_360.csv"
        self.assertRaises(urllib.error.HTTPError, lambda: read_file(url))


if __name__ == '__main__':
    unittest.main()
