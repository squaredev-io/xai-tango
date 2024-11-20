import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_data(data):
    """
    Preprocess the uploaded dataset by:
    - Cleaning column names.
    - Filling missing values for specific columns.
    - Hashing high-cardinality features.
    - Encoding categorical variables.
    - Applying TF-IDF on text data.
    - Extracting datetime features from timestamps.
    - Converting columns to numeric where possible.
    """
    # Check and clean column names
    data.columns = data.columns.str.strip()  # Remove spaces
    data.columns = data.columns.str.lower()  # Convert to lowercase

    # Fill missing values with appropriate placeholders
    data.fillna({
        "id_recipient_subsidiary": "UNKNOWN",
        "id_recipient_institute": "UNKNOWN",
        "useragent": "UNKNOWN",
        "id_sender_subsidiary": "UNKNOWN"
    }, inplace=True)

    # Hash high-cardinality fields (e.g., IBANs)
    high_cardinality_features = [
        "id_recipient_account",
        "id_sender_account"
    ]
    for feature in high_cardinality_features:
        data[feature] = data[feature].apply(lambda x: hash(x) % 10**8)  # Hashing to numeric values

    # Encode categorical features
    categorical_features = [
        "id_recipient_country",
        "id_recipient_subsidiary",
        "id_recipient_institute",
        "abroad",
        "sepa",
        "channel",
        "instant",
        "useragent",
        "currency"
    ]
    label_encoders = {}
    for feature in categorical_features:
        le = LabelEncoder()
        data[feature] = le.fit_transform(data[feature].astype(str))
        label_encoders[feature] = le

    # Apply TF-IDF on 'Causal'
    if "causal" in data.columns:
        tfidf = TfidfVectorizer(max_features=100)
        causal_features = tfidf.fit_transform(data["causal"].astype(str)).toarray()

        # Create a DataFrame for TF-IDF features and merge it
        causal_df = pd.DataFrame(causal_features, columns=[f"causal_tfidf_{i}" for i in range(causal_features.shape[1])])
        data = pd.concat([data.reset_index(drop=True), causal_df], axis=1)
        data.drop(columns=["causal"], inplace=True)

    # Convert timestamp to datetime features
    if "timestamp" in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'], format="%Y/%m/%d %H:%M:%S", errors="coerce")
        data['hour'] = data['timestamp'].dt.hour
        data['day'] = data['timestamp'].dt.day
        data['month'] = data['timestamp'].dt.month
        data.drop(columns=["timestamp"], inplace=True)  # Drop original timestamp

    data = data.drop(columns=["id_recipient_account"])
    # Convert all columns to numeric where possible
    def ensure_numeric(df):
        non_numeric_columns = []
        for column in df.columns:
            try:
                # Try to convert to numeric
                pd.to_numeric(df[column])
            except ValueError:
                # Keep track of non-numeric columns
                non_numeric_columns.append(column)
        # Drop non-numeric columns
        df = df.drop(columns=non_numeric_columns)
        return df, non_numeric_columns

    data, dropped_columns = ensure_numeric(data)
    
    return data
