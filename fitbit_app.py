import streamlit as st
import pandas as pd
from preprocessing import preprocess_data
from regression import run_regression
from clustering import run_clustering

st.title("Fitbit Project: Calorie Burn Prediction & Workout Pattern Clustering")

# Load dataset
data = pd.read_csv("cleaned_Fitbit_dataset.csv")

# Normalize column names
data.columns = data.columns.str.strip().str.replace(" ", "_").str.lower()

st.subheader("Dataset Preview")
st.write(data.head())
st.write("Dataset Columns:", data.columns.tolist())

# Preprocess
data_encoded = preprocess_data(data)

# Regression
st.header("Task 1: calorie Burn Prediction")
run_regression(data_encoded)

# Clustering
st.header("Task 2: Workout Pattern Clustering")
run_clustering(data_encoded)


