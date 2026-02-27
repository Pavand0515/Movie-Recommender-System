import pandas as pd
import ast

def load_data():
    movies = pd.read_csv("data/tmdb_5000_movies.csv")
    credits = pd.read_csv("data/tmdb_5000_credits.csv")
    movies = movies.merge(credits, on="title")
    return movies

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

def get_top_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

def preprocess(movies):
    movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

    movies.dropna(inplace=True)

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(get_top_cast)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

    return movies[['movie_id','title','tags']]import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import load_data, preprocess

movies = load_data()
movies = preprocess(movies)

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

similarity = cosine_similarity(vectors)

pickle.dump(movies, open("artifacts/movies.pkl", "wb"))
pickle.dump(similarity, open("artifacts/similarity.pkl", "wb"))

print("Model")


