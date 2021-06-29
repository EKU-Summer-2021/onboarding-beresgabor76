import pandas as pd


def read_file(url):
    """Reads in a csv file to a DataFrame from a specified url"""
    df = pd.read_csv(url, header=None)
    return df


def print_summary(df):
    """Prints out the statistics of a DataFrame"""
    print(df.describe())


