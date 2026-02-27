import pickle

movies = pickle.load(open("artifacts/movies.pkl", "rb"))
similarity = pickle.load(open("artifacts/similarity.pkl", "rb"))

def recommend(movie, top_n=5):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)),
                        reverse=True,
                        key=lambda x: x[1])[1:top_n+1]

    recommendations = []
    for i in movie_list:
        recommendations.append(movies.iloc[i[0]].title)

    return recommendations
