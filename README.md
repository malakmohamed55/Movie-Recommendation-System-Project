# 🎬 Movie Recommendation System

A hybrid movie recommendation system built using Content-Based Filtering, Collaborative Filtering, and a Web App powered by Streamlit. This project demonstrates how to personalize movie suggestions based on user preferences and past behavior, similar to Netflix and IMDb's recommender systems.

---

## 📌 Project Features

- ✅ Content-Based Filtering (TF-IDF + Cosine Similarity)
- ✅ Collaborative Filtering (SVD via Surprise library)
- ✅ Hybrid Recommendation Engine (combines both approaches)
- ✅ Streamlit Web App for interactive recommendations
- ✅ Cleaned and preprocessed movie and rating data
- ✅ Modular, readable code and notebook organization

---

## 🧠 Recommendation Techniques

### 1. Content-Based Filtering
- Uses movie genres and textual features
- Computes similarity between movies
- Great for new or inactive users

### 2. Collaborative Filtering
- Based on user-item interaction matrix
- Learns patterns in user behavior
- Uses Matrix Factorization (SVD)

### 3. Hybrid Engine
- Combines both models
- Improves accuracy and diversity
- Handles cold-start problems more effectively

---

## 🗂️ Project Structure
Movie-Recommendation-System/
│
├── data/ # Raw data files
│ ├── movies.csv
│ └── ratings.csv
│
├── notebooks/ # Jupyter notebooks for modeling
│ ├── Data Preprocessing.ipynb
│ ├── Content-Based Filtering.ipynb
│ ├── Collaborative Filtering.ipynb
│ └── Hybrid Recommendation Engine.ipynb
│
├── app/
│ └── streamlit_app.py # Streamlit web application
│
├── requirements.txt # Python dependencies
├── README.md # This file

streamlit run app/streamlit_app.py
Datasets Used
MovieLens Dataset (Small 100K)

movies.csv – contains movie titles and genres

ratings.csv – user ratings for movies

Evaluation Metrics
Metric	Description
RMSE	Accuracy of predicted ratings
Precision@K	% of relevant movies in top-K recommendations
Recall@K	% of relevant movies retrieved
Coverage	% of items that can be recommended
Diversity	Variation in recommended items


 Technologies Used
Python 🐍

Pandas, NumPy

Scikit-learn

Surprise (for SVD)

Streamlit (Web UI)

Jupyter Notebooks
