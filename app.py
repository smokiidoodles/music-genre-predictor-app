"""
app.py

Streamlit application to predict music genre using a pre-trained ensemble model.
All inputs are sliders representing realistic ranges of audio features.
"""
import streamlit as st
import pandas as pd
import numpy as np
import random
from joblib import load

# -------------------------
# Load model and label encoder
# -------------------------
model = load("model.joblib")
le = load("label_encoder.joblib")

# -------------------------
# Feature names used in training
# -------------------------
feature_names = [
    'Unnamed: 0', 'popularity', 'danceability', 'energy',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

# -------------------------
# Load reference songs for recommendations
# -------------------------
song_ref = pd.read_csv("song_reference.csv")
feature_cols = ['popularity','danceability','energy','acousticness',
                'instrumentalness','liveness','valence','tempo']

# -------------------------
# App UI
# -------------------------
st.title("🎵 Music Genre Predictor & Recommender")
st.write("Enter audio features below:")

# Random ID
unnamed_0 = random.randint(0, 11200)

# Input sliders
popularity = st.slider("Popularity", 0, 100, 50)
danceability = st.slider("Danceability", 0.0, 1.0, 0.5, 0.01)
energy = st.slider("Energy", 0.0, 1.0, 0.5, 0.01)
acousticness = st.slider("Acousticness", 0.0, 0.01, 0.0005, 0.000001, format="%.8f")
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0, 0.000001, format="%.8f")
liveness = st.slider("Liveness", 0.01, 0.9, 0.05, 0.01, format="%.4f")
valence = st.slider("Valence", 0.0, 1.0, 0.5, 0.01)
tempo = st.slider("Tempo (BPM)", 60, 200, 120, 1)

# Build input dataframe
input_data = pd.DataFrame([[
    unnamed_0, popularity, danceability, energy,
    acousticness, instrumentalness, liveness, valence, tempo
]], columns=feature_names)

# -------------------------
# Predict Genre Button
# -------------------------
if st.button("Predict Genre"):
    pred_class = model.predict(input_data)[0]
    genre = le.inverse_transform([pred_class])[0]
    st.success(f"🎧 Predicted Genre: {genre}")

# -------------------------
# Recommend Songs Button
# -------------------------
if st.button("Recommend Songs"):
    # Build feature array for comparison (exclude ID)
    user_features = np.array([[popularity, danceability, energy,
                               acousticness, instrumentalness,
                               liveness, valence, tempo]])

    # Compute Euclidean distance
    distances = np.linalg.norm(song_ref[feature_cols].values - user_features, axis=1)
    song_ref['distance'] = distances

    # Get top 5 closest songs
    top_songs = song_ref.nsmallest(5, 'distance')[['track_name','artists']]
    st.subheader("🎵 Recommended Songs")
    st.table(top_songs)