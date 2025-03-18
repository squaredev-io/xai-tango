# from lime.lime_tabular import LimeTabularExplainer
# import shap
# import matplotlib.pyplot as plt


# def lime_explainer(model, X_train, X_test, selected_row_index, class_names=None):
#     """
#     Generate Lime explanations for a given instance in the dataset.

#     Args:
#         model: Trained machine learning model.
#         X_train: Preprocessed training dataset (Pandas DataFrame).
#         X_test: Preprocessed test dataset (Pandas DataFrame).
#         selected_row_index: Index of the instance in X_test to explain.
#         class_names: List of class names (e.g., ["Not Fraud", "Fraud"]).

#     Returns:
#         Lime explanation object.
#     """
#     # Align columns in X_test with X_train
#     X_test_aligned = X_test[X_train.columns]

#     # Initialize the LIME explainer
#     explainer = LimeTabularExplainer(
#         training_data=X_train.values,  # Training data as a numpy array
#         feature_names=X_train.columns,  # Column names
#         class_names=class_names,  # Class names (if provided)
#         mode="classification",  # Specify classification mode
#     )

#     # Select an instance to explain
#     instance = X_test_aligned.iloc[selected_row_index]
#     # Generate explanation for the selected instance
#     explanation = explainer.explain_instance(
#         data_row=instance.values,  # The instance to explain
#         predict_fn=model.predict_proba,  # Prediction function
#         num_features=10,  # Number of features to show
#     )
#     return explanation


# def shap_explainer(model, data):
#     """
#     Generate Shap explanations for the entire dataset.
#     Args:
#         model: Trained machine learning model.
#         data: Preprocessed dataset (Pandas DataFrame).
#     Returns:
#         Shap explanation object.
#     """
#     # Generate SHAP values
#     explainer = shap.Explainer(model)
#     shap_values = explainer(data)

#     # Clear previous plots to avoid double plots
#     plt.clf()

#     # Customize the summary plot
#     plt.figure(figsize=(12, 8))  # Adjust figure size
#     shap.summary_plot(shap_values, data, show=False)
#     plt.xticks(fontsize=10)  # Adjust font size
#     plt.yticks(fontsize=10)
#     plt.tight_layout()  # Fix layout to avoid overlapping
#     plt.show()  # Ensure only the current plot is shown

#     return shap_values

from lime.lime_tabular import LimeTabularExplainer
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class KerasModelWrapper:
    def __init__(self, model):
        self.model = model
        self.expected_input_dim = model.input_shape[-1]

    def predict_proba(self, X):
        # Ensure input is a NumPy array
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Check shape before prediction
        if X.shape[1] != self.expected_input_dim:
            raise ValueError(
                f"[KerasModelWrapper] Invalid input shape: expected {self.expected_input_dim} features, got {X.shape[1]}"
            )

        preds = self.model.predict(X)

        # Handle binary classification
        if preds.ndim == 1 or preds.shape[1] == 1:
            return np.hstack([1 - preds, preds])

        return preds


def lime_explainer(model, X_train, X_test, selected_row_index, class_names=None, keras_model=False):
    """
    Generate Lime explanations for a given instance in the dataset.

    Args:
        model: Trained machine learning model (Keras or sklearn).
        X_train: Preprocessed training dataset (Pandas DataFrame).
        X_test: Preprocessed test dataset (Pandas DataFrame).
        selected_row_index: Index of the instance in X_test to explain.
        class_names: List of class names.
        keras_model: Set True if the model is Keras-based.
    Returns:
        Lime explanation object.
    """
    X_test_aligned = X_test[X_train.columns]

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=class_names,
        mode="classification",
    )

    instance = X_test_aligned.iloc[selected_row_index]

    if keras_model:
        model = KerasModelWrapper(model)

    explanation = explainer.explain_instance(data_row=instance.values, predict_fn=model.predict_proba, num_features=10)
    return explanation


def shap_explainer(model, data, keras_model=False):
    """
    Generate Shap explanations for the entire dataset.
    Args:
        model: Trained machine learning model.
        data: Preprocessed dataset (Pandas DataFrame).
        keras_model: Set True if using a Keras model.
    Returns:
        Shap explanation object.
    """
    # For Keras, use a specific masker and DeepExplainer or GradientExplainer
    if keras_model:
        explainer = shap.Explainer(model, data)
    else:
        explainer = shap.Explainer(model)

    shap_values = explainer(data)

    plt.clf()
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, data, show=False)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

    return shap_values
