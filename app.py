"""
app.py

Streamlit application to predict music genre using a pre-trained ensemble model.
All inputs are sliders representing realistic ranges of audio features.
"""

import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# -------------------------------------------------
# Load artifacts
# -------------------------------------------------
model = load("model.joblib")
le = load("label_encoder.joblib")
FEATURE_COLUMNS = load("feature_columns.joblib")
scaler = load("scaler.joblib")

song_ref = pd.read_csv("song_reference.csv")

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("🎵 Music Genre Predictor")

st.write("Adjust the audio features:")

popularity = st.slider("Popularity", 0, 100, 50)
danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.1)
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
liveness = st.slider("Liveness", 0.0, 1.0, 0.1)
valence = st.slider("Valence", 0.0, 1.0, 0.5)
tempo = st.slider("Tempo (BPM)", 60, 200, 120)

# -------------------------------------------------
# Hidden defaults (NOT shown to user)
# -------------------------------------------------
loudness = -8.0
duration_ms = 180000
speechiness = 0.05

# -------------------------------------------------
# Build input
# -------------------------------------------------
input_df = pd.DataFrame([{
    'popularity': popularity,
    'danceability': danceability,
    'energy': energy,
    'acousticness': acousticness,
    'instrumentalness': instrumentalness,
    'liveness': liveness,
    'valence': valence,
    'tempo': tempo,
    'loudness': loudness,
    'duration_ms': duration_ms,
    'speechiness': speechiness
}])

# Feature engineering
input_df['energy_acoustic_ratio'] = input_df['energy'] / (input_df['acousticness'] + 1e-5)
input_df['loud_instr'] = input_df['loudness'] * input_df['instrumentalness']
input_df['tempo_bin'] = pd.cut(input_df['tempo'], bins=3, labels=[0,1,2])
input_df['duration_bin'] = pd.cut(input_df['duration_ms'], bins=3, labels=[0,1,2])

# Reorder columns
input_df = input_df[FEATURE_COLUMNS]

# Scale
input_scaled = scaler.transform(input_df)



# -------------------------------------------------
# Recommendations
# -------------------------------------------------
if st.button("Recommend Similar Songs"):
    user_vec = np.array([[popularity, danceability, energy,
                           acousticness, instrumentalness,
                           liveness, valence, tempo]])

    features = ['popularity','danceability','energy','acousticness',
                'instrumentalness','liveness','valence','tempo']

    distances = np.linalg.norm(song_ref[features].values - user_vec, axis=1)
    song_ref['distance'] = distances

    top = song_ref.nsmallest(5, 'distance')[['track_name','artists']]
    st.subheader("🎵 Recommended Songs")
    st.table(top)
