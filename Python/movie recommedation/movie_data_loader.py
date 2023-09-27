import pandas as pd


def load_movie_full():
    # read movielens.org datasets
    #
    print('Loading tags.csv ... ')
    tags = pd.read_csv("tags.csv")
    print('End')
    print('Loading ratings.csv ... ')
    ratings = pd.read_csv("ratings.csv")
    print('End')
    print('Loading links.csv ... ')
    links = pd.read_csv("links.csv", dtype={'imdbId': str, 'tmdbId': str})
    links['imdbId'] = links['imdbId'].astype(str)
    links['tmdbId'] = links['tmdbId'].astype(str)
    print('End')
    print('Loading movies.csv ... ')
    movies = pd.read_csv("movies.csv")
    print('End')
    print('Loading genome-tags.csv ... ')
    genome_tags = pd.read_csv("genome-tags.csv")
    print('End')
    print('Loading genome-scores.csv ... ')
    genome_scores = pd.read_csv("genome-scores.csv")
    print('End')

    # compute some on-off data
    #
    print('Aggregating ratings by movieId ... ')
    ratings_agg = ratings.groupby('movieId').mean().round(2)
    print('End')
    print('Merging movies and aggregated ratings')
    movie_rating = pd.merge(movies, ratings_agg, how='left', on=['movieId'])
    print('End')

    print('Merging movie ratings with links ... ')
    movie_rating_link = pd.merge(movie_rating, links, how='left', on=['movieId'])
    print('End')

    print('MOVIE_FULL merge ... ')
    movie_full = pd.merge(movie_rating_link, tags, how='left', on=['movieId'])
    print('End')

    print('MOVIE_FULL groupby ... ')
    movie_full = movie_full.groupby(['title', 'tag'], as_index=False).max()
    print('End')

    print('MOVIE_FULL saving t0 csv ... ')
    movie_full.to_csv('movie_full.csv')
    print('End')

    return movie_full


if __name__ == '__main__':
    load_movie_full()
