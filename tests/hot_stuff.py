import unittest
from src import HotStuff


class TestHotStuff(unittest.TestCase):
    def setUp(self):
        self.__hot_stuff = HotStuff()

    def test_weeks_of_number_one_hit(self):
        #given
        hit_song = 'One Sweet Day'
        EXPECTED = 16
        #when
        ACTUAL = self.__hot_stuff.weeks_of_number_one_hit(hit_song)
        #then
        self.assertEqual(EXPECTED, ACTUAL)

    def test_songs_hit_podium_but_not_no1(self):
        #given
        EXPECTED = 614
        #when
        ACTUAL = self.__hot_stuff.songs_hit_podium_but_not_no1()
        # then
        self.assertEqual(EXPECTED, ACTUAL)

    def test_artists_with_the_same_song(self):
        #given
        hit_song = "Don't Just Stand There"
        EXPECTED = 1
        #when
        ACTUAL = self.__hot_stuff.artists_with_the_same_song(hit_song)
        # then
        self.assertEqual(EXPECTED, ACTUAL)

    def test_song_on_the_chart_most_weeks(self):
        #given
        EXPECTED = 'Radioactive'
        #when
        ACTUAL = self.__hot_stuff.song_on_the_chart_most_weeks()
        #then
        self.assertEqual(EXPECTED, ACTUAL)

    def test_artist_with_the_most_song(self):
        #given
        EXPECTED = 'Glee Cast'
        #when
        ACTUAL = self.__hot_stuff.artist_with_the_most_song()
        #then
        self.assertEqual(EXPECTED, ACTUAL)

    def test_artist_with_the_most_no1_weeks(self):
        #given
        EXPECTED = 'Mariah Carey'
        #when
        ACTUAL = self.__hot_stuff.artist_with_the_most_no1_weeks()
        #then
        self.assertEqual(EXPECTED, ACTUAL)


if __name__ == '__main__':
    unittest.main()
