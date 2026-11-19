# ================================================================
# PROJECT 2: SPOTIFY SONGS' GENRE SEGMENTATION
# Complete End-to-End Machine Learning Project
# ================================================================


# ================================================================
# 1. IMPORT LIBRARIES
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

sns.set_style("whitegrid")


# ================================================================
# 2. LOAD DATASET
# ================================================================

# Make sure Project 2.csv is in the same folder as this notebook
df = pd.read_csv("Project 2.csv")

print("=" * 70)
print("SPOTIFY SONGS' GENRE SEGMENTATION")
print("=" * 70)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# ================================================================
# 3. BASIC INFORMATION
# ================================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nDataset Information:")
print(df.info())

print("\nNumber of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])

print("\nColumn Names:")
print(df.columns.tolist())


# ================================================================
# 4. MISSING VALUES
# ================================================================

print("\n" + "=" * 70)
print("MISSING VALUE ANALYSIS")
print("=" * 70)

missing = df.isnull().sum()

missing_percentage = (
    df.isnull().sum() / len(df)
) * 100

missing_table = pd.DataFrame({
    "Missing Values": missing,
    "Percentage": missing_percentage
})

print(missing_table.sort_values(
    "Percentage",
    ascending=False
))


# ================================================================
# 5. REMOVE DUPLICATES
# ================================================================

print("\nDuplicates before:", df.duplicated().sum())

df = df.drop_duplicates()

print("Duplicates after:", df.duplicated().sum())


# ================================================================
# 6. HANDLE MISSING VALUES
# ================================================================

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

numerical_columns = df.select_dtypes(
    include=np.number
).columns

# Fill categorical missing values with mode
for col in categorical_columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(
            df[col].mode()[0]
        )

# Fill numerical missing values with median
for col in numerical_columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(
            df[col].median()
        )

print("\nTotal missing values after preprocessing:",
      df.isnull().sum().sum())


# ================================================================
# 7. DESCRIPTIVE STATISTICS
# ================================================================

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

print(df.describe())


# ================================================================
# 8. DEFINE AUDIO FEATURES
# ================================================================

audio_features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]


# ================================================================
# 9. GENRE DISTRIBUTION
# ================================================================

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="playlist_genre",
    order=df["playlist_genre"].value_counts().index
)

plt.title("Distribution of Songs Across Genres")
plt.xlabel("Playlist Genre")
plt.ylabel("Number of Songs")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ================================================================
# 10. SUBGENRE DISTRIBUTION
# ================================================================

top_subgenres = (
    df["playlist_subgenre"]
    .value_counts()
    .head(20)
)

plt.figure(figsize=(12, 8))

sns.barplot(
    x=top_subgenres.values,
    y=top_subgenres.index
)

plt.title("Top 20 Playlist Subgenres")
plt.xlabel("Number of Songs")
plt.ylabel("Subgenre")
plt.tight_layout()
plt.show()


# ================================================================
# 11. TOP PLAYLISTS
# ================================================================

top_playlists = (
    df["playlist_name"]
    .value_counts()
    .head(20)
)

plt.figure(figsize=(12, 8))

sns.barplot(
    x=top_playlists.values,
    y=top_playlists.index
)

plt.title("Top 20 Playlists by Number of Songs")
plt.xlabel("Number of Songs")
plt.ylabel("Playlist")
plt.tight_layout()
plt.show()


# ================================================================
# 12. TOP ARTISTS
# ================================================================

top_artists = (
    df["track_artist"]
    .value_counts()
    .head(20)
)

plt.figure(figsize=(12, 8))

sns.barplot(
    x=top_artists.values,
    y=top_artists.index
)

plt.title("Top 20 Artists by Number of Songs")
plt.xlabel("Number of Songs")
plt.ylabel("Artist")
plt.tight_layout()
plt.show()


# ================================================================
# 13. TRACK POPULARITY DISTRIBUTION
# ================================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["track_popularity"],
    bins=30,
    kde=True
)

plt.title("Distribution of Track Popularity")
plt.xlabel("Popularity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# ================================================================
# 14. AUDIO FEATURE DISTRIBUTIONS
# ================================================================

for feature in audio_features:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df[feature],
        bins=30,
        kde=True
    )

    plt.title(
        f"Distribution of {feature}"
    )

    plt.xlabel(feature)
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


# ================================================================
# 15. BOX PLOTS FOR OUTLIER DETECTION
# ================================================================

for feature in audio_features:

    plt.figure(figsize=(8, 4))

    sns.boxplot(
        x=df[feature]
    )

    plt.title(
        f"Boxplot of {feature}"
    )

    plt.xlabel(feature)

    plt.tight_layout()
    plt.show()


# ================================================================
# 16. DANCEABILITY BY GENRE
# ================================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="playlist_genre",
    y="danceability"
)

plt.title("Danceability Across Genres")
plt.xlabel("Genre")
plt.ylabel("Danceability")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ================================================================
# 17. ENERGY BY GENRE
# ================================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="playlist_genre",
    y="energy"
)

plt.title("Energy Across Genres")
plt.xlabel("Genre")
plt.ylabel("Energy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ================================================================
# 18. VALENCE BY GENRE
# ================================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="playlist_genre",
    y="valence"
)

plt.title("Valence Across Genres")
plt.xlabel("Genre")
plt.ylabel("Valence")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ================================================================
# 19. AVERAGE AUDIO FEATURES BY GENRE
# ================================================================

genre_features = (
    df.groupby("playlist_genre")[audio_features]
    .mean()
)

print("\nAverage Audio Features by Genre:")
print(genre_features)


# ================================================================
# 20. HEATMAP OF GENRE AUDIO FEATURES
# ================================================================

plt.figure(figsize=(14, 7))

sns.heatmap(
    genre_features,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title(
    "Average Audio Features by Genre"
)

plt.tight_layout()
plt.show()


# ================================================================
# 21. CORRELATION MATRIX
# ================================================================

correlation_features = (
    audio_features +
    ["track_popularity"]
)

correlation = df[
    correlation_features
].corr()

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title(
    "Correlation Matrix of Spotify Audio Features"
)

plt.tight_layout()
plt.show()


# ================================================================
# 22. STRONGEST CORRELATIONS
# ================================================================

corr_pairs = correlation.unstack()

corr_pairs = corr_pairs[
    corr_pairs != 1
].sort_values(
    ascending=False
)

print("\nTop Positive Correlations:")
print(corr_pairs.head(15))

print("\nTop Negative Correlations:")
print(corr_pairs.tail(15))


# ================================================================
# 23. ENERGY VS LOUDNESS
# ================================================================

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="energy",
    y="loudness",
    hue="playlist_genre",
    alpha=0.5
)

plt.title("Energy vs Loudness")
plt.tight_layout()
plt.show()


# ================================================================
# 24. DANCEABILITY VS VALENCE
# ================================================================

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="danceability",
    y="valence",
    hue="playlist_genre",
    alpha=0.5
)

plt.title("Danceability vs Valence")
plt.tight_layout()
plt.show()


# ================================================================
# 25. ENERGY VS DANCEABILITY
# ================================================================

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="energy",
    y="danceability",
    hue="playlist_genre",
    alpha=0.5
)

plt.title("Energy vs Danceability")
plt.tight_layout()
plt.show()


# ================================================================
# 26. PREPARE DATA FOR CLUSTERING
# ================================================================

cluster_features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

X_cluster = df[
    cluster_features
].copy()


# ================================================================
# 27. STANDARDIZATION
# ================================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X_cluster
)

print("\nScaled Data Shape:")
print(X_scaled.shape)


# ================================================================
# 28. ELBOW METHOD
# ================================================================

inertia = []

K = range(2, 11)

for k in K:

    kmeans_temp = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans_temp.fit(X_scaled)

    inertia.append(
        kmeans_temp.inertia_
    )


plt.figure(figsize=(8, 5))

plt.plot(
    K,
    inertia,
    marker="o"
)

plt.title(
    "Elbow Method for Optimal Number of Clusters"
)

plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")

plt.tight_layout()
plt.show()


# ================================================================
# 29. SILHOUETTE SCORE
# ================================================================

silhouette_scores = []

for k in range(2, 11):

    kmeans_temp = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels_temp = (
        kmeans_temp.fit_predict(X_scaled)
    )

    score = silhouette_score(
        X_scaled,
        labels_temp
    )

    silhouette_scores.append(score)


plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 11),
    silhouette_scores,
    marker="o"
)

plt.title(
    "Silhouette Score for Different K Values"
)

plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")

plt.tight_layout()
plt.show()


# ================================================================
# 30. SELECT BEST K USING SILHOUETTE SCORE
# ================================================================

best_k = (
    range(2, 11)
)[
    np.argmax(silhouette_scores)
]

print("\nBest number of clusters according to Silhouette Score:",
      best_k)


# ================================================================
# 31. BUILD FINAL K-MEANS MODEL
# ================================================================

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["cluster"] = (
    kmeans.fit_predict(X_scaled)
)

print("\nCluster Distribution:")
print(
    df["cluster"]
    .value_counts()
    .sort_index()
)


# ================================================================
# 32. PCA FOR 2D VISUALIZATION
# ================================================================

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]


print("\nPCA Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

print(
    "Total Variance Explained:",
    sum(pca.explained_variance_ratio_)
)


# ================================================================
# 33. CLUSTER VISUALIZATION
# ================================================================

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=df,
    x="PCA1",
    y="PCA2",
    hue="cluster",
    palette="tab10",
    alpha=0.6
)

plt.title(
    "Spotify Songs Clustering using K-Means"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.tight_layout()
plt.show()


# ================================================================
# 34. CLUSTER VS GENRE
# ================================================================

cluster_genre = pd.crosstab(
    df["cluster"],
    df["playlist_genre"]
)

print("\nCluster vs Genre:")
print(cluster_genre)


plt.figure(figsize=(10, 6))

sns.heatmap(
    cluster_genre,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "Cluster vs Playlist Genre"
)

plt.tight_layout()
plt.show()


# ================================================================
# 35. CLUSTER VS POPULAR PLAYLISTS
# ================================================================

top_playlist_names = (
    df["playlist_name"]
    .value_counts()
    .head(15)
    .index
)

filtered_df = df[
    df["playlist_name"]
    .isin(top_playlist_names)
]

plt.figure(figsize=(14, 7))

sns.countplot(
    data=filtered_df,
    x="playlist_name",
    hue="cluster"
)

plt.title(
    "Clusters Across Popular Playlists"
)

plt.xticks(
    rotation=60,
    ha="right"
)

plt.tight_layout()
plt.show()


# ================================================================
# 36. CLUSTER CHARACTERISTICS
# ================================================================

cluster_summary = (
    df.groupby("cluster")[cluster_features]
    .mean()
)

print("\nCluster Characteristics:")
print(cluster_summary)


plt.figure(figsize=(14, 7))

sns.heatmap(
    cluster_summary,
    annot=True,
    fmt=".2f",
    cmap="viridis"
)

plt.title(
    "Average Audio Characteristics of Each Cluster"
)

plt.tight_layout()
plt.show()


# ================================================================
# 37. DOMINANT GENRE IN EACH CLUSTER
# ================================================================

dominant_genre = (
    df.groupby("cluster")["playlist_genre"]
    .agg(
        lambda x:
        x.value_counts().index[0]
    )
)

print("\nDominant Genre in Each Cluster:")

for cluster, genre in dominant_genre.items():

    print(
        f"Cluster {cluster}: {genre}"
    )


# ================================================================
# 38. GENRE CLASSIFICATION MODEL
# ================================================================

X = df[
    cluster_features
]

y = df[
    "playlist_genre"
]


# ================================================================
# 39. TRAIN TEST SPLIT
# ================================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)


# ================================================================
# 40. SCALE CLASSIFICATION FEATURES
# ================================================================

scaler_classifier = StandardScaler()

X_train_scaled = (
    scaler_classifier
    .fit_transform(X_train)
)

X_test_scaled = (
    scaler_classifier
    .transform(X_test)
)


# ================================================================
# 41. RANDOM FOREST CLASSIFIER
# ================================================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train_scaled,
    y_train
)


# ================================================================
# 42. PREDICTION
# ================================================================

y_pred = rf_model.predict(
    X_test_scaled
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 70)
print("GENRE CLASSIFICATION RESULTS")
print("=" * 70)

print(
    "\nRandom Forest Accuracy:",
    accuracy
)


# ================================================================
# 43. CLASSIFICATION REPORT
# ================================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ================================================================
# 44. CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(10, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=rf_model.classes_,
    yticklabels=rf_model.classes_
)

plt.title(
    "Genre Classification Confusion Matrix"
)

plt.xlabel("Predicted Genre")
plt.ylabel("Actual Genre")

plt.tight_layout()
plt.show()


# ================================================================
# 45. FEATURE IMPORTANCE
# ================================================================

importance = pd.Series(
    rf_model.feature_importances_,
    index=cluster_features
).sort_values(
    ascending=False
)

print("\nFeature Importance:")
print(importance)


plt.figure(figsize=(10, 6))

sns.barplot(
    x=importance.values,
    y=importance.index
)

plt.title(
    "Feature Importance for Genre Prediction"
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()
plt.show()


# ================================================================
# 46. RECOMMENDATION SYSTEM
# ================================================================

recommendation_features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

song_features = df[
    recommendation_features
]

scaler_rec = StandardScaler()

song_features_scaled = (
    scaler_rec.fit_transform(
        song_features
    )
)


# ================================================================
# 47. COSINE SIMILARITY
# ================================================================

similarity_matrix = cosine_similarity(
    song_features_scaled
)


# ================================================================
# 48. SONG RECOMMENDATION FUNCTION
# ================================================================

def recommend_songs(
    song_name,
    number_of_recommendations=5
):

    # Find song
    matches = df[
        df["track_name"]
        .str.lower()
        == song_name.lower()
    ]

    if matches.empty:

        return (
            "Song not found in dataset."
        )

    song_index = matches.index[0]

    # Get similarity scores
    similarity_scores = list(
        enumerate(
            similarity_matrix[
                song_index
            ]
        )
    )

    # Sort from highest similarity
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Remove original song
    similarity_scores = (
        similarity_scores[1:]
    )

    # Select recommended songs
    recommended_indices = [
        index
        for index, score
        in similarity_scores[
            :number_of_recommendations
        ]
    ]

    recommendations = df.loc[
        recommended_indices,
        [
            "track_name",
            "track_artist",
            "playlist_genre",
            "playlist_subgenre",
            "track_popularity"
        ]
    ]

    return recommendations


# ================================================================
# 49. IMPROVED GENRE-BASED RECOMMENDATION
# ================================================================

def recommend_by_song(
    song_name,
    n=10
):

    matches = df[
        df["track_name"]
        .str.lower()
        == song_name.lower()
    ]

    if matches.empty:

        return "Song not found."

    idx = matches.index[0]

    selected_genre = df.loc[
        idx,
        "playlist_genre"
    ]

    # Only consider songs from same genre
    candidates = df[
        df["playlist_genre"]
        == selected_genre
    ].copy()

    candidate_indices = (
        candidates.index
    )

    scores = similarity_matrix[
        idx,
        candidate_indices
    ]

    candidates[
        "similarity"
    ] = scores

    result = (
        candidates
        .sort_values(
            "similarity",
            ascending=False
        )
    )

    # Remove original song
    result = result[
        result.index != idx
    ]

    result = result.head(n)

    return result[
        [
            "track_name",
            "track_artist",
            "playlist_genre",
            "playlist_subgenre",
            "track_popularity",
            "similarity"
        ]
    ]


# ================================================================
# 50. DISPLAY SAMPLE SONGS FOR TESTING
# ================================================================

print("\n" + "=" * 70)
print("SAMPLE SONGS AVAILABLE FOR RECOMMENDATION")
print("=" * 70)

print(
    df[
        [
            "track_name",
            "track_artist",
            "playlist_genre"
        ]
    ].head(20)
)


# ================================================================
# 51. TEST RECOMMENDATION SYSTEM
# ================================================================

sample_song = df.iloc[0]["track_name"]

print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

print(
    "\nInput Song:",
    sample_song
)

print(
    "\nRecommended Songs:"
)

print(
    recommend_songs(
        sample_song,
        5
    )
)


# ================================================================
# 52. TEST IMPROVED RECOMMENDATION SYSTEM
# ================================================================

print("\n" + "=" * 70)
print("GENRE-BASED RECOMMENDATIONS")
print("=" * 70)

print(
    recommend_by_song(
        sample_song,
        10
    )
)


# ================================================================
# 53. FINAL PROJECT SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("FINAL PROJECT SUMMARY")
print("=" * 70)

print(
    "\nOriginal Dataset Shape:",
    df.shape
)

print(
    "\nNumber of Genres:",
    df["playlist_genre"].nunique()
)

print(
    "\nNumber of Subgenres:",
    df["playlist_subgenre"].nunique()
)

print(
    "\nNumber of Clusters:",
    df["cluster"].nunique()
)

print(
    "\nBest K:",
    best_k
)

print(
    "\nSilhouette Score:",
    max(silhouette_scores)
)

print(
    "\nRandom Forest Accuracy:",
    accuracy
)

print(
    "\nRecommendation System: READY"
)


# ================================================================
# 54. SAVE FINAL DATASET
# ================================================================

df.to_csv(
    "spotify_genre_segmented.csv",
    index=False
)

print(
    "\nFinal segmented dataset saved as:"
    " spotify_genre_segmented.csv"
)

print("\nPROJECT COMPLETED SUCCESSFULLY!")