
import pandas as pd


movies = pd.read_csv('ml-20m/movies.csv')

def loadMovieNames():
    movieNames = {}
    for index, field in movies.iterrows():
        movieNames[int(field['movieId'])] = field['title']
    return movieNames
nameDict = loadMovieNames()