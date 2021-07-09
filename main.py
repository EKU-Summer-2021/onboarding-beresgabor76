import numpy as np
from src import Polynomial
from src.file_reader import read_file, print_summary
from src import HotStuff

if __name__ == '__main__':
    coeffs = np.array([1,0,0])
    polynom = Polynomial(coeffs)
    print(polynom.evaluate(3))
    print(polynom.roots())
    df = read_file("https://raw.githubusercontent.com/EKU-Summer-2021/intelligent_system_data/" +
                   "main/Intelligent%20System%20Data/CSP/CSP_360.csv")
    print_summary(df)

    print('\nAnalyzing hot_stuff.csv data file')
    hot_stuff = HotStuff()
    #Task1
    hit_song = 'One Sweet Day'
    number_one_weeks = hot_stuff.weeks_of_number_one_hit(hit_song)
    print(hit_song + ' hit week position No.1: ' + str(number_one_weeks) + ' times')
    #Task2
    songs_number_two_three = hot_stuff.songs_hit_podium_but_not_no1()
    print("Number of songs hit podium but didn't reach No.1: " + str(songs_number_two_three))
    #Task3
    hit_song = "Don't Just Stand There"
    number_of_artists = hot_stuff.artists_with_the_same_song(hit_song)
    print("Number of artists with song name " + hit_song + ": " + str(number_of_artists))
    #Task4
    hit_song = hot_stuff.song_on_the_chart_most_weeks()
    print("The song hit the chart most weeks: " + hit_song)
    #Task5
    artist = hot_stuff.artist_with_the_most_song()
    print("The artist with the most different songs hit the chart: " + artist)

