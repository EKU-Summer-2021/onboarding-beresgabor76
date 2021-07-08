"""
Module for analyzing data in hot_stuff.csv file
"""
import os
import pandas as pd


class HotStuff:
    """
    Class for storing and analyzing data in hot_stuff.csv file
    """
    def __init__(self):
        self.__filepath = os.path.join(os.path.dirname(__file__), '../data', 'hot_stuff.csv')
        self.__data = pd.read_csv(self.__filepath)

    def weeks_of_number_one_hit(self, song):
        """
        Returns the number of weeks while the given song was hit No.1
        """
        return self.__data.loc[(self.__data['Song'] == song) &
                               (self.__data['Week Position'] == 1), 'Week Position'].count()

    def songs_hit_podium_but_not_no1(self):
        """
        Returns the number of songs that hit the podium but didn't reach No.1
        """
        return self.__data.groupby('SongID')[['SongID', 'Peak Position']]\
            .filter(lambda data: ((data['Peak Position'] == 2).all() | (data['Peak Position'] == 3).all()))\
            .count()['SongID']

    def artists_with_the_same_song(self, song):
        """
        Returns the number of artists that have the same given song name
        """
        return self.__data.groupby('Performer')[['Performer', 'Song']]\
            .filter(lambda data: ((data['Song'] == song).any()))['Performer']\
            .nunique()
