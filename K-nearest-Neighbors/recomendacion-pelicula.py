import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# Cargar datasets
users = pd.read_csv('users.csv')
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Crear matriz de usuarios vs películas
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Calcular similitud de usuarios
similarity_matrix = cosine_similarity(user_movie_matrix)

# Implementar KNN para encontrar usuarios similares
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(user_movie_matrix)
distances, indices = knn.kneighbors(user_movie_matrix, n_neighbors=5)

# Función para predecir la calificación de una película
def predict_rating(user_index, movie_id):
    similar_users = indices[user_index]
    similar_users_ratings = user_movie_matrix.iloc[similar_users, movie_id]
    pred_rating = similar_users_ratings.mean()
    return pred_rating

# Recomendación de películas
user_id = 10  # Ejemplo de usuario
user_index = user_id - 1  # Ajustar índice
movie_id = 50  # Ejemplo de película
predicted_rating = predict_rating(user_index, movie_id)
print(f"Predicted rating for movie {movie_id} by user {user_id}: {predicted_rating}")
