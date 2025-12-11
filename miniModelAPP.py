from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, make_scorer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from joblib import dump
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pickle

# Loading dataset
d = pd.read_csv("dataset_copy.csv", skiprows=0)

# Removed columns not used for modeling
non_featured_cols = ['track_name', 'track_id', 'explicit', 'artists', 'key','album_name', 'mode',
                     'speechiness', 'liveness','valence', 'time_signature']

data = d.drop(non_featured_cols, axis='columns')

# Processing track_genre column
data['track_genre'] = data['track_genre'].apply(
    lambda x: x[0] if isinstance(x, list) else x
)

data['track_genre'] = data['track_genre'].fillna('unknown').str.strip().str.lower()

# Encode genre
le = LabelEncoder()
data['track_genre_encoded'] = le.fit_transform(data['track_genre'])
pickle.dump(le,open("label_encoder.pkl", "wb"))

# Save the LabelEncoder
dump(le, "label_encoder.joblib", compress=3)

# Features and target
value_to_drop = 'world-music'
data = data.drop(data[data['track_genre'] == value_to_drop].index)
X = data.drop(columns=['track_genre', 'track_genre_encoded'])
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
y = data['track_genre_encoded']

# -----------------------------------------------------
# 2. PROPORTIONAL STRATIFIED SAMPLING
# -----------------------------------------------------
frac = 0.20     # keep 20% of rows per genre

sampled_data = data.groupby('track_genre_encoded', group_keys=False).apply(
    lambda x: x.sample(frac=frac)
).reset_index(drop=True)

print("Original size:", len(data))
print("Sampled size:", len(sampled_data))
# ---------------------------

# -----------------------------------------------------
# 4. TRAIN / TEST SPLIT with STRATIFICATION
# -----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)


#Model Testing Begins Below
clf = RandomForestClassifier(
    n_estimators=50,
    max_depth=15,
    class_weight="balanced"
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('SET UP CROSS VALIDATION')

from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=5, test_size=0.2)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X, y, cv=cv)
print()
print("Cross fold validation accuracy scores:",scores)
print("Cross validation accuracy mean:",scores.mean())


#Random Forest
print("NEXT DATA TEST-Random Forest")
rf_scores = cross_validate(clf, X, y, cv=5,scoring={
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted')
    }
)

print("Random Forest - Accuracy:", np.mean(rf_scores['test_accuracy']))
print("Random Forest - Precision:", np.mean(rf_scores['test_precision']))
print("Predicted genre names (first 10):", le.inverse_transform(y_pred[:10]))

#Gaussian Nave Bayes
print("FIRST DATA PREDICTION")
clf1 = GaussianNB()
gnb_scores = cross_validate( clf1, X, y, cv=5,scoring={'accuracy': make_scorer(accuracy_score),
             'precision': make_scorer(precision_score, average='weighted')
    }
)

print("GaussianNB - Accuracy:", np.mean(gnb_scores['test_accuracy']))
print("GaussianNB - Precision:", np.mean(gnb_scores['test_precision']))


#Decision Tree


print('NEXT DATA PREDICTION- Decision Tree')
clf2 = DecisionTreeClassifier()

dt_scores = cross_validate( clf2, X, y, cv=5,scoring={'accuracy': make_scorer(accuracy_score),
             'precision': make_scorer(precision_score, average='weighted')
    }
)

print("Decision Tree- Accuracy:", np.mean(dt_scores['test_accuracy']))
print("Decision Tree - Precision:", np.mean(dt_scores['test_precision']))


#Complex Classifier Below
print('NEXT DATA PREDICTION- VOTING(ENSEMBLE LEARNING)')
eclf1 = VotingClassifier(estimators=[ ('RF', clf), ('GNB', clf1), ('p', clf2)], voting='soft')
eclf1 = eclf1.fit(X,y)


# Save the trained model (RandomForest / ensemble)
dump(eclf1, "model.joblib", compress=5)

scores2 = cross_val_score(eclf1, X, y, cv=cv)
print()
print("Cross fold validation accuracy scores for the ensemble:",scores2)
print("Cross fold validation accuracy mean for the ensemble:",scores2.mean())

scores3 = cross_val_score(eclf1, X, y, cv=cv, scoring='precision_macro')
print()
print("Cross fold validation precision scores for the ensemble:",scores3)
print("Cross fold validation precision mean for the ensemble:",scores3.mean())