import streamlit as st
import pickle
import pandas as pd
from xai_banking.utils.data_processing import cached_preprocess_data
from xai_banking.utils.explainers import lime_explainer, shap_explainer
from xai_banking.utils.utils import *
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split

def main():

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

    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("Use Default Data"):
            st.session_state["model"] = pickle.load(open(DEFAULT_MODEL_PATH, "rb"))
            st.session_state["data"] = pd.read_csv(DEFAULT_DATA_PATH)
            st.success("Default model and dataset loaded successfully!")

    with col2:
        if st.button("Reset Session"):
            for key in ["model", "data", "processed_data", "X_train", "X_test", "y_train", "y_test"]:
                st.session_state[key] = None
            st.success("Session has been reset.")

    # Data Preprocessing and Splitting
    if st.session_state["model"] and st.session_state["data"] is not None:
        # Ensure processed_data is created if not already done
        if st.session_state["processed_data"] is None:
            st.session_state["processed_data"] = cached_preprocess_data(st.session_state["data"])

        # Display processed data safely
        if st.session_state["processed_data"] is not None:
            with st.expander("Preprocessed Dataset"):
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
    create_lime_vs_shap_info()
    method = st.selectbox("Choose Explainability Method", ["Lime", "Shap"])

    # Row Selection for Lime (Only Display When "Lime" is Selected)
    selected_row_index = None
    if method == "Lime" and st.session_state["X_test"] is not None:
        st.subheader("Select Row for Explanation")
        selected_row_index = st.number_input(
            "Select Row Index",
            min_value=0,
            max_value=len(st.session_state["X_test"]) - 1,
            value=0,
            step=1
        )
        st.info(f"Selected row: {selected_row_index}")

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

        if method == "Shap" and st.session_state["model"] and st.session_state["X_test"] is not None:
            descriptions = load_descriptions()

            # Store SHAP values in session state to avoid recomputation
            if "shap_values" not in st.session_state:
                st.session_state["shap_values"] = shap_explainer(
                    st.session_state["model"],
                    st.session_state["X_test"]
                )
            shap_values = st.session_state["shap_values"]

            st.header("SHAP Explanation")
            
            # 1. SHAP Summary Plot
            st.subheader("Summary Plot")
            with st.expander(descriptions["summary_plot"]["title"]):
                display_description("summary_plot", descriptions)
            plt.clf()
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, st.session_state["X_test"], show=False)
            plt.tight_layout()
            st.pyplot(plt.gcf())

            # 2. SHAP Bar Plot
            st.subheader("Bar Plot")
            with st.expander(descriptions["bar_plot"]["title"]):
                display_description("bar_plot", descriptions)
            plt.clf()
            plt.figure(figsize=(10, 6))
            shap.plots.bar(shap_values)
            plt.tight_layout()
            st.pyplot(plt.gcf())

    # Waterfall Plot Section (Handled Separately to Avoid Reruns)
    if method == "Shap" and st.session_state["model"] and st.session_state["X_test"] is not None:
        shap_values = st.session_state.get("shap_values")  # Fetch stored SHAP values
        descriptions = load_descriptions()
        if shap_values:
            # 1. SHAP Summary Plot
            st.subheader("Summary Plot")
            with st.expander(descriptions["summary_plot"]["title"]):
                display_description("summary_plot", descriptions)
            plt.clf()
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, st.session_state["X_test"], show=False)
            plt.tight_layout()
            st.pyplot(plt.gcf())

            # 2. SHAP Bar Plot
            st.subheader("Bar Plot")
            with st.expander(descriptions["bar_plot"]["title"]):
                display_description("bar_plot", descriptions)
            plt.clf()
            plt.figure(figsize=(10, 6))
            shap.plots.bar(shap_values)
            plt.tight_layout()
            st.pyplot(plt.gcf())

            # 3. Watrerfall Plot
            st.subheader("Waterfall Plot")
            with st.expander(descriptions["waterfall_plot"]["title"]):
                display_description("waterfall_plot", descriptions)

            # Add a number_input for selecting the data point
            selected_instance = st.number_input(
                "Select Data Point for Waterfall Plot",
                min_value=0,
                max_value=len(st.session_state["X_test"]) - 1,
                value=0,
                step=1,
                key="waterfall_instance"
            )
            st.info(f"Displaying Waterfall Plot for Data Point: {selected_instance}")

            # Dynamically display the selected instance's Waterfall Plot
            plt.clf()
            shap.plots.waterfall(shap_values[selected_instance])
            plt.tight_layout()
            st.pyplot(plt.gcf())

            # 4. SHAP Heatmap Plot
            st.subheader("Heatmap Plot")
            with st.expander(descriptions["heatmap_plot"]["title"]):
                display_description("heatmap_plot", descriptions)
            plt.clf()
            plt.figure(figsize=(12, 8))
            shap.plots.heatmap(shap_values)
            plt.tight_layout()
            st.pyplot(plt.gcf())

            # 5. SHAP Beeswarm Plot
            st.subheader("Beeswarm Plot")
            with st.expander(descriptions["beeswarm_plot"]["title"]):
                display_description("beeswarm_plot", descriptions)
            plt.clf()
            plt.figure(figsize=(12, 8))
            shap.plots.beeswarm(shap_values, max_display=20)
            plt.tight_layout()
            st.pyplot(plt.gcf())

        else:
            st.error("Please upload both model and dataset to proceed.")