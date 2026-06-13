import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

def run_clustering(data_encoded):
    # Normalize column names
    data_encoded.columns = data_encoded.columns.str.strip().str.replace(" ", "_").str.lower()

    # Choose the correct target column
    target_col = "calories_burned_(kcal)"   # or "calories_burned_(kcal)_log"

    if target_col not in data_encoded.columns:
        st.error(f"No '{target_col}' column found. Available columns: {data_encoded.columns.tolist()}")
        return

    # Drop target column for clustering
    X_cluster = data_encoded.drop(target_col, axis=1)
    scaler = StandardScaler()
    X_cluster_scaled = scaler.fit_transform(X_cluster)

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_cluster_scaled)

    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_pca)

    sil_score = silhouette_score(X_pca, clusters)
    st.write(f"**Silhouette Score:** {sil_score:.2f}")

    # Plot clusters
    fig, ax = plt.subplots()
    ax.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap="viridis", alpha=0.7)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("Workout Clusters")
    st.pyplot(fig)

    # Cluster summary
    cluster_summary = pd.DataFrame(X_cluster)
