"""
miniModelAPP.py

This script trains multiple classification models (Random Forest, GaussianNB, Decision Tree)
on a sampled subset of the music genre dataset. It then creates an ensemble VotingClassifier
and saves both the trained model and the label encoder for deployment in a Streamlit app.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from joblib import dump


# Load dataset
# -------------------------
data = pd.read_csv("dataset_copy.csv")


# -------------------------
# Data for Reccomendations
# -------------------------
recommend_df = data[[
    'track_name', 'artists', 'popularity', 'danceability', 'energy',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]]

# Save for use in Streamlit
recommend_df.to_csv("song_reference.csv", index=False)

# -------------------------
# Data for Training
# -------------------------

# Removed irrelevant columns
non_featured_cols = [
    'track_name', 'track_id', 'explicit', 'artists', 'key', 'album_name',
    'mode', 'speechiness', 'time_signature', 'loudness', 'duration_ms'
]
data = data.drop(non_featured_cols, axis=1)

# -------------------------
# Processed target column
# -------------------------

# Flatten track_genre if stored as a list
data['track_genre'] = data['track_genre'].apply(lambda x: x[0] if isinstance(x, list) else x)
data['track_genre'] = data['track_genre'].fillna('unknown').str.strip().str.lower()

# Encode genre
le = LabelEncoder()
data['track_genre_encoded'] = le.fit_transform(data['track_genre'])

# Saved label encoder for deployment
dump(le, "label_encoder.joblib", compress=3)

# Removed unwanted genres
data = data[data['track_genre'] != 'world-music']

# -------------------------
# Proportional stratified sampling (reduce dataset size)
# -------------------------
sample_frac = 0.10  # 10% of each genre
sampled_data = data.groupby('track_genre_encoded', group_keys=False).apply(
    lambda x: x.sample(frac=sample_frac)
).reset_index(drop=True)

print(f"Original size: {len(data)}, Sampled size: {len(sampled_data)}")

X = sampled_data.drop(columns=['track_genre', 'track_genre_encoded'])
y = sampled_data['track_genre_encoded']

# -------------------------
# Train/test split with stratification
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# -------------------------
# Initialization of classifiers
# -------------------------
rf = RandomForestClassifier(n_estimators=50, max_depth=50, class_weight="balanced")
gnb = GaussianNB()
dt = DecisionTreeClassifier()

# Train Random Forest
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# -------------------------
# Cross-validation setup
# -------------------------
cv = ShuffleSplit(n_splits=5, test_size=0.2)

# Random Forest
rf_scores = cross_validate(
    rf, X, y, cv=5,
    scoring={
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted')
    }
)
print("Random Forest - Accuracy:", np.mean(rf_scores['test_accuracy']))
print("Random Forest - Precision:", np.mean(rf_scores['test_precision']))

# GaussianNB
gnb_scores = cross_validate(
    gnb, X, y, cv=5,
    scoring={
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted')
    }
)
print("GaussianNB - Accuracy:", np.mean(gnb_scores['test_accuracy']))
print("GaussianNB - Precision:", np.mean(gnb_scores['test_precision']))

# Decision Tree
dt_scores = cross_validate(
    dt, X, y, cv=5,
    scoring={
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted')
    }
)
print("Decision Tree - Accuracy:", np.mean(dt_scores['test_accuracy']))
print("Decision Tree - Precision:", np.mean(dt_scores['test_precision']))

# -------------------------
# Ensemble: Voting Classifier (soft)
# -------------------------
ensemble = VotingClassifier(estimators=[('RF', rf), ('GNB', gnb), ('DT', dt)], voting='soft')
ensemble.fit(X, y)

# Save ensemble model
dump(ensemble, "model.joblib", compress=5)

# Ensemble cross-validation
ensemble_scores_acc = cross_validate(ensemble, X, y, cv=cv, scoring='accuracy')
ensemble_scores_prec = cross_validate(ensemble, X, y, cv=cv, scoring='precision_macro')

print("\nEnsemble - Accuracy mean:", np.mean(ensemble_scores_acc['test_score']))
print("Ensemble - Precision mean:", np.mean(ensemble_scores_prec['test_score']))

# Print first 10 ensemble predictions
y_pred_ensemble = ensemble.predict(X_test)
print("\nFIRST 10 ENSEMBLE PREDICTIONS:")
for i in range(10):
    true_label = le.inverse_transform([y_test.iloc[i]])[0]
    pred_label = le.inverse_transform([y_pred_ensemble[i]])[0]
    print(f"Sample {i+1}: True = {true_label} | Predicted = {pred_label}")
