import streamlit as st
import pickle
import numpy as np
import random
from joblib import load


# Loading your saved model + label encoder

model = load("model.joblib")
le = load("label_encoder.joblib")

# ----------------------------
# Streamlit App UI
# ----------------------------
st.title("🎵 Music Genre Predictor")
st.write("Enter the audio features below to predict the music genre.")

#input
unnamed_0 = random.randint(0, 1000000)  # can be any number
popularity = st.slider("Popularity", 0, 100, 50)
duration_ms = st.number_input("Duration (ms)", 10000, 600000, 180000)
danceability = st.slider("Danceability", 0.0, 1.0, 0.5, 0.01)
energy = st.slider("Energy", 0.0, 1.0, 0.5, 0.01)
loudness = st.number_input("Loudness (dB)", -60.0, 5.0, -10.0, 0.1)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5, 0.01)
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0, 0.01)
tempo = st.number_input("Tempo (BPM)", 40.0, 250.0, 120.0, 1.0)

input_data = np.array([[unnamed_0, popularity, duration_ms, danceability, energy,
    loudness, acousticness, instrumentalness, tempo]])


# Prediction button
if st.button("Predict Genre"):


    # Predict class index
    pred = model.predict(input_data)[0]

    # Convert index to class name
    genre = le.inverse_transform([pred])[0]

    st.success(f"🎧 **Predicted Genre:** {genre}")

