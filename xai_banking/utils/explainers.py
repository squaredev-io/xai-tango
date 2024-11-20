from lime.lime_tabular import LimeTabularExplainer
import shap

def lime_explainer(model, X_train, X_test, selected_row_index, class_names=None):
    """
    Generate Lime explanations for a given instance in the dataset.

    Args:
        model: Trained machine learning model.
        X_train: Preprocessed training dataset (Pandas DataFrame).
        X_test: Preprocessed test dataset (Pandas DataFrame).
        selected_row_index: Index of the instance in X_test to explain.
        class_names: List of class names (e.g., ["Not Fraud", "Fraud"]).
    
    Returns:
        Lime explanation object.
    """
    # Align columns in X_test with X_train
    X_test_aligned = X_test[X_train.columns]

    # Initialize the LIME explainer
    explainer = LimeTabularExplainer(
        training_data=X_train.values,         # Training data as a numpy array
        feature_names=X_train.columns,        # Column names
        class_names=class_names,              # Class names (if provided)
        mode="classification"                 # Specify classification mode
    )

    # Select an instance to explain
    instance = X_test_aligned.iloc[selected_row_index]
    print((X_test_aligned), "ddd")
    # Generate explanation for the selected instance
    explanation = explainer.explain_instance(
        data_row=instance.values,             # The instance to explain
        predict_fn=model.predict_proba,       # Prediction function
        num_features=10,                       # Number of features to show
    )
    return explanation


def shap_explainer(model, data):
    """
    Generate Shap explanations for the entire dataset.
    Args:
        model: Trained machine learning model.
        data: Preprocessed dataset (Pandas DataFrame).
    Returns: 
        Shap explanation object.
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(data)
    return shap_values
