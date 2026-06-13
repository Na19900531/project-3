from sklearn.preprocessing import LabelEncoder

def preprocess_data(data):
    # Normalize column names
    data.columns = data.columns.str.strip().str.replace(" ", "_").str.lower()

    # Identify categorical columns (object/string types)
    categorical_cols = data.select_dtypes(include=["object"]).columns

    data_encoded = data.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        data_encoded[col] = le.fit_transform(data_encoded[col].astype(str))

    return data_encoded





