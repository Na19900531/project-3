import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def run_regression(data_encoded):
    # Normalize column names
    data_encoded.columns = data_encoded.columns.str.strip().str.replace(" ", "_").str.lower()

    # Choose the correct target column
    target_col = "calories_burned_(kcal)"   # or "calories_burned_(kcal)_log"

    if target_col not in data_encoded.columns:
        st.error(f"No '{target_col}' column found. Available columns: {data_encoded.columns.tolist()}")
        return

    # Split features and target
    X = data_encoded.drop(target_col, axis=1)
    y = data_encoded[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Choose model
    model_choice = st.selectbox("Choose Regression Model", ["Linear Regression", "Random Forest"])
    if model_choice == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(random_state=42)

    # Train model
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    # Plot Actual vs Predicted
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
    ax.set_xlabel("Actual Calories Burned (kcal)")
    ax.set_ylabel("Predicted Calories Burned (kcal)")
    ax.set_title("Actual vs Predicted Calories")
    st.pyplot(fig)


