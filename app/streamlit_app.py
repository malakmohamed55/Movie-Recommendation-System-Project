
import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------- Load data ----------- #
@st.cache_data
def load_data():
    movies= pd.read_csv(r"C:\Users\Admin\OneDrive\Documents\Desktop\final project ip\movies.csv")   
    ratings = pd.read_csv(r"C:\Users\Admin\OneDrive\Documents\Desktop\final project ip\ratings.csv")  
    df = pd.merge(ratings, movies, on="movieId").drop(columns=["timestamp"]).drop_duplicates()
    return ratings, movies, df

ratings, movies, df = load_data()
movies_df = df[['movieId', 'title', 'genres']].drop_duplicates(subset='movieId')

# ----------- Content-based similarity ----------- #
@st.cache_resource
def build_content_model():
    tfidf = TfidfVectorizer(stop_words='english')
    movies_df['genres'] = movies_df['genres'].fillna('')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = build_content_model()
indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()

# ----------- Collaborative model ----------- #
@st.cache_resource
def train_svd():
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    model = SVD()
    model.fit(trainset)
    return model

svd_model = train_svd()

# ----------- Hybrid recommendations ----------- #
def hybrid_recommendations(user_id, movie_title, content_weight=0.5, top_n=10):
    idx = indices.get(movie_title, None)
    if idx is None:
        return ["Movie not found."]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:100]

    hybrid_scores = []
    for i, content_score in sim_scores:
        movie_id = movies_df.iloc[i]['movieId']
        try:
            collab_score = svd_model.predict(user_id, movie_id).est
        except:
            collab_score = 3.0  # fallback
        
        final_score = content_weight * content_score + (1 - content_weight) * (collab_score / 5)
        hybrid_scores.append((movie_id, final_score))

    top_movies = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:top_n]
    recommendations = [movies_df[movies_df['movieId'] == mid]['title'].values[0] for mid, _ in top_movies]
    return recommendations

# ----------- Streamlit UI ----------- #
st.title("🎬 Hybrid Movie Recommender System")


user_name = st.text_input("Enter your name:")
user_id = st.number_input("Enter your user ID:", min_value=1, step=1)

movie_title = st.selectbox("Choose a movie you like:", sorted(movies_df['title'].unique()))
content_weight = st.slider("Content-based Weight (Collaborative = 1 - Weight):", 0.0, 1.0, 0.5)

if st.button("Get Recommendations"):
    st.subheader(f"🎥 Top Recommendations for {user_name or 'User'}:")
    results = hybrid_recommendations(user_id, movie_title, content_weight)
    for i, movie in enumerate(results, 1):
        st.write(f"{i}. {movie}")
