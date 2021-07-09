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
            .filter(lambda data: ((data['Peak Position'] == 2).all()
                                  | (data['Peak Position'] == 3).all()))\
            .count()['SongID']

    def artists_with_the_same_song(self, song):
        """
        Returns the number of artists that have the same given song name
        """
        return self.__data.groupby('Performer')[['Performer', 'Song']]\
            .filter(lambda data: ((data['Song'] == song).any()))['Performer']\
            .nunique()

    def song_on_the_chart_most_weeks(self):
        """
        Returns the song name that hit the chart the most weeks
        """
        max_weeks = self.__data.groupby('SongID').count().max()[0]
        hits = self.__data.groupby(['SongID', 'Song'])['SongID'].count()
        for hit in list(hits.items()):
            if hit[1] == max_weeks:
                return hit[0][1]

    def artist_with_the_most_song(self):
        """
        Returns the artist that has the most different songs that hit the chart
        """
        performers = self.__data.groupby(['Performer'])['Performer'].nunique()
        performers -= 1
        hits = self.__data.groupby(['Performer', 'SongID'])['SongID']
        for (hit, value) in hits:
            performers[performers.index == hit[0]] += 1
        max_songs = performers.max()
        return performers[performers == max_songs].index[0]

    def artist_with_the_most_no1_weeks(self):
        """
        Returns the performer who hit the chart at No.1 the most weeks
        """
        hits = self.__data.loc[self.__data['Week Position'] == 1,
                               ['Performer', 'SongID', 'Week Position']]
        performers_hits = hits.groupby(['Performer'])['Performer'].count()
        max_performers_hits = performers_hits.max()
        return performers_hits[performers_hits == max_performers_hits].index[0]
