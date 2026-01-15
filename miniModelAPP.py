

"""
miniModelAPP.py

Trains an ensemble classifier to predict music genre (4 classes)
and saves:
- trained model
- label encoder
- feature column list
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, make_scorer, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from joblib import dump

# -------------------------------------------------
# Load dataset
# -------------------------------------------------
data = pd.read_csv("dataset_copy.csv", index_col=0)

# -------------------------------------------------
# Save reference songs for recommendations
# -------------------------------------------------
recommend_df = data[[
    'track_name', 'artists', 'popularity', 'danceability', 'energy',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]]
recommend_df.to_csv("song_reference.csv", index=False)

# -------------------------------------------------
# Drop non-feature columns
# -------------------------------------------------
non_featured_cols = [
    'track_name','track_id','explicit','artists','key',
    'album_name','mode','time_signature'
]
data = data.drop(columns=non_featured_cols, errors='ignore')

# -------------------------------------------------
# Process track_genre
# -------------------------------------------------
data['track_genre'] = data['track_genre'].apply(
    lambda x: x[0] if isinstance(x, list) else x
)
data['track_genre'] = data['track_genre'].fillna('unknown').str.lower().str.strip()

# -------------------------------------------------
# Collapse to 4 genres
# -------------------------------------------------
def map_genre_4(g):
    if any(k in g for k in ['pop','k-pop','j-pop','c-pop','electro']):
        return 'pop'
    if any(k in g for k in ['rock','alt','indie','garage','classic','metal']):
        return 'rock'
    if any(k in g for k in ['hip hop','rap','trap']):
        return 'hiphop'
    if any(k in g for k in ['jazz','blues','r&b','soul','ambient','classical','chill']):
        return 'jazz_soothing'
    return None

data['genre_grouped'] = data['track_genre'].apply(map_genre_4)
data = data.dropna(subset=['genre_grouped'])

# -------------------------------------------------
# Encode target
# -------------------------------------------------
le = LabelEncoder()
data['genre_encoded'] = le.fit_transform(data['genre_grouped'])

# -------------------------------------------------
# Features & Target
# -------------------------------------------------
X = data.drop(columns=['track_genre','genre_grouped','genre_encoded'])
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
y = data['genre_encoded']

# -------------------------------------------------
# Feature Engineering
# -------------------------------------------------
X['energy_acoustic_ratio'] = X['energy'] / (X['acousticness'] + 1e-5)
X['loud_instr'] = X['loudness'] * X['instrumentalness']
X['tempo_bin'] = pd.cut(X['tempo'], bins=3, labels=[0,1,2])
X['duration_bin'] = pd.cut(X['duration_ms'], bins=3, labels=[0,1,2])

# -------------------------------------------------
# Train / Test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

# -------------------------------------------------
# Scaling (needed for KNN)
# -------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------
# Models
# -------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=40,
    min_samples_leaf=3,
    class_weight='balanced'
)

dt = DecisionTreeClassifier(class_weight='balanced')
knn = KNeighborsClassifier(n_neighbors=7)

# -------------------------------------------------
# Ensemble
# -------------------------------------------------
ensemble = VotingClassifier(
    estimators=[('RF', rf), ('DT', dt), ('KNN', knn)],
    voting='soft'
)

ensemble.fit(X_train_scaled, y_train)

# -------------------------------------------------
# Save artifacts
# -------------------------------------------------
dump(ensemble, "model.joblib", compress=5)
dump(le, "label_encoder.joblib", compress=3)
dump(X.columns.tolist(), "feature_columns.joblib", compress=3)
dump(scaler, "scaler.joblib", compress=3)



