# 🎵 Spotify Songs Genre Segmentation Using Machine Learning

## 📌 Project Overview

Music streaming platforms like Spotify contain millions of songs belonging to different genres and musical styles. Organizing songs based on their musical characteristics can help improve music discovery and recommendation systems.

This project focuses on **Spotify Songs Genre Segmentation using Unsupervised Machine Learning**. It uses Spotify audio features to group songs into different clusters based on similarities in their musical characteristics.

The project applies data preprocessing, Exploratory Data Analysis (EDA), feature scaling, and clustering algorithms to discover hidden patterns in music data.

## 🎯 Objectives

- Analyze Spotify songs based on their audio features.
- Perform data cleaning and preprocessing.
- Understand relationships between different musical attributes.
- Apply Unsupervised Machine Learning techniques.
- Group similar songs using clustering algorithms.
- Visualize and interpret the generated clusters.
- Explore the potential of clustering for music recommendation systems.

## 📂 Dataset Description

The project uses a Spotify songs dataset containing information about songs and their audio characteristics.

Important features may include:

| Feature | Description |
|---|---|
| Danceability | Measures how suitable a track is for dancing |
| Energy | Represents the intensity and activity of a track |
| Loudness | Overall loudness of a track |
| Acousticness | Measures the acoustic characteristics of a track |
| Instrumentalness | Predicts whether a track contains vocals |
| Valence | Describes the musical positivity of a track |
| Tempo | Measures the speed of a track in BPM |
| Speechiness | Detects the presence of spoken words |
| Duration | Total duration of the song |

*Note: The exact features depend on the dataset used in this project.*

## 🛠️ Technologies & Libraries Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Jupyter Notebook | Development Environment |
| Pandas | Data Manipulation |
| NumPy | Numerical Computation |
| Matplotlib | Data Visualization |
| Seaborn | Statistical Visualization |
| Scikit-learn | Machine Learning |

## ⚙️ Project Workflow

### 1. Data Collection
Importing the Spotify songs dataset for analysis and segmentation.

### 2. Data Cleaning and Preprocessing
- Checking for missing values.
- Identifying and handling duplicate records.
- Selecting relevant numerical audio features.
- Removing irrelevant columns where necessary.
- Applying feature scaling to normalize numerical values.

### 3. Exploratory Data Analysis (EDA)
- Analyzing the distribution of Spotify audio features.
- Creating correlation heatmaps.
- Visualizing relationships between musical attributes.
- Identifying patterns in song characteristics.

### 4. Feature Selection
Selecting relevant audio features such as danceability, energy, acousticness, valence, loudness, and tempo for clustering.

### 5. Model Building
Applying clustering algorithms to group songs based on similarities in their audio features.

**K-Means Clustering:** Used to divide songs into different clusters based on their musical characteristics.

### 6. Cluster Evaluation
Evaluating clustering performance using:
- Elbow Method
- Silhouette Score
- Davies-Bouldin Index
- Within-Cluster Sum of Squares (WCSS)

### 7. Visualization
Visualizing the generated clusters to understand the similarities and differences between groups of songs.

## 📊 Results

The project successfully demonstrates the use of Unsupervised Machine Learning for grouping Spotify songs based on their audio characteristics.

### Key Findings

- Identified patterns and relationships among different Spotify audio features.
- Grouped songs based on similarities in their musical characteristics.
- Used clustering techniques to discover hidden structures in the dataset.
- Visualized clusters to understand the distribution of songs.
- Explored how clustering can support music discovery and recommendation systems.

### Model Performance

| Evaluation Metric | Result |
|---|---|
| Number of Clusters | To be updated |
| Silhouette Score | To be updated |
| Davies-Bouldin Index | To be updated |
| Inertia (WCSS) | To be updated |

*Actual values should be updated based on the model output.*

## 💡 Applications

- Music Recommendation Systems
- Automated Playlist Generation
- Music Discovery
- Similarity-Based Song Grouping
- Music Data Analysis

## 💻 Installation and Usage

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Navigate to the Project Directory

```bash
cd Spotify-Songs-Genre-Segmentation
```

### Step 3: Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Step 4: Launch Jupyter Notebook

```bash
jupyter notebook
```

### Step 5: Run the Project

Open the project notebook and execute the cells sequentially to perform data preprocessing, exploratory data analysis, clustering, and evaluation.

## 🔮 Future Improvements

- Experiment with Hierarchical Clustering and DBSCAN.
- Apply Principal Component Analysis (PCA) for dimensionality reduction.
- Improve cluster quality through hyperparameter tuning.
- Develop an interactive music recommendation system.
- Create an automated playlist generator based on audio similarities.

## 🏁 Conclusion

This project demonstrates the practical application of Unsupervised Machine Learning in music data analysis.

By analyzing Spotify audio features and applying clustering techniques, the project identifies groups of songs with similar musical characteristics without relying on predefined genre labels.

It provides valuable hands-on experience in data preprocessing, feature engineering, clustering algorithms, and visualization while exploring the potential of Machine Learning in music recommendation systems.

## 👩‍💻 Author

**Shivanshi Karanjai**

CSE – Artificial Intelligence and Machine Learning

GitHub: [@soulfulshivu](https://github.com/soulfulshivu)

---

⭐ If you find this project interesting, consider giving this repository a star!
