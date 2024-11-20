import streamlit as st
import pickle
import pandas as pd
from utils.data_processing import preprocess_data
from utils.explainers import lime_explainer, shap_explainer
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split

# Default file paths
DEFAULT_MODEL_PATH = "xai_banking/utils/random_forest_model.pkl"
DEFAULT_DATA_PATH = "xai_banking/utils/synthetic_dataset.csv"

# Initialize session state
if "model" not in st.session_state:
    st.session_state["model"] = None

if "data" not in st.session_state:
    st.session_state["data"] = None

if "processed_data" not in st.session_state:
    st.session_state["processed_data"] = None

if "X_train" not in st.session_state:
    st.session_state["X_train"] = None

if "X_test" not in st.session_state:
    st.session_state["X_test"] = None

if "y_train" not in st.session_state:
    st.session_state["y_train"] = None

if "y_test" not in st.session_state:
    st.session_state["y_test"] = None

# Streamlit App Title
st.title("Model Explainability Tool")
st.sidebar.header("Upload Model and Dataset")

# Reset Button
if st.sidebar.button("Reset Session"):
    for key in ["model", "data", "processed_data", "X_train", "X_test", "y_train", "y_test"]:
        st.session_state[key] = None
    st.sidebar.success("Session has been reset.")

# Model Upload
model_path = st.sidebar.file_uploader("Upload Model (.pkl)", type=["pkl"])
if model_path:
    st.session_state["model"] = pickle.load(model_path) 
    st.sidebar.success("Model loaded successfully!")

# Dataset Upload
data_path = st.sidebar.file_uploader("Upload Dataset (.csv)", type=["csv"])
if data_path:
    st.session_state["data"] = pd.read_csv(data_path)
    st.sidebar.success("Dataset loaded successfully!")

# Use Default Data Button
if st.sidebar.button("Use Default Data"):
    st.session_state["model"] = pickle.load(open(DEFAULT_MODEL_PATH, "rb"))
    st.session_state["data"] = pd.read_csv(DEFAULT_DATA_PATH)
    st.sidebar.success("Default model and dataset loaded successfully!")

# Data Preprocessing and Splitting
if st.session_state["model"] and st.session_state["data"] is not None:
    st.session_state["processed_data"] = preprocess_data(st.session_state["data"])
    st.write("Preprocessed Dataset:")
    st.dataframe(st.session_state["processed_data"].head())

    # Splitting data into training and testing sets
    X = st.session_state["processed_data"].drop(columns=["label_fraud_post"])
    y = st.session_state["processed_data"]["label_fraud_post"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test

# Explainability Method Selection
method = st.selectbox("Choose Explainability Method", ["Lime", "Shap"])

# Row Selection for Lime
selected_row_index = None
if st.session_state["X_test"] is not None:
    st.sidebar.header("Select Row for Explanation")
    selected_row_index = st.sidebar.number_input(
        "Select Row Index",
        min_value=0,
        max_value=len(st.session_state["X_test"]) - 1,
        value=0,
        step=1
    )
    st.sidebar.info(f"Selected row: {selected_row_index}")

# Generate Explanations
if st.button("Generate Explanations"):
    if method == "Lime" and st.session_state["model"] and st.session_state["X_train"] is not None:
        st.subheader("Lime Explanation")
        explanation = lime_explainer(
            st.session_state["model"],
            st.session_state["X_train"],
            st.session_state["X_test"],
            selected_row_index,
            class_names=["Not Fraud", "Fraud"]
        )
        
        st.write("Explanation as Text:")
        st.text(explanation.as_list())
        st.pyplot(explanation.as_pyplot_figure())
    elif method == "Shap" and st.session_state["model"] and st.session_state["processed_data"] is not None:
        st.subheader("Shap Explanation")
        shap_values = shap_explainer(
            st.session_state["model"],
            st.session_state["X_test"] 
        )
        
        shap.summary_plot(shap_values, st.session_state["X_test"], show=False)
        st.pyplot(plt.gcf())

        shap.plots.bar(shap_values)
        st.pyplot(plt.gcf())

        shap.plots.waterfall(shap_values[0])
        st.pyplot(plt.gcf())

        shap.plots.heatmap(shap_values)
        st.pyplot(plt.gcf())

        shap.plots.beeswarm(shap_values)
        st.pyplot(plt.gcf())

    else:
        st.error("Please upload both model and dataset to proceed.")
