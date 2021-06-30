"""
File reading module Issue#2
"""

import pandas as pd


def read_file(url):
    """Reads in a csv file to a DataFrame from a specified url"""
    result_df = pd.read_csv(url, header=None)
    return result_df


def print_summary(summarized_df):
    """Prints out the statistics of a DataFrame"""
    print(summarized_df.describe())
