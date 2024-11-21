import matplotlib.pyplot as plt
import shap
import streamlit as st
import json

def shap_beeswarm_plot(shap_values, data):
    """
    Generate a beeswarm plot for SHAP values.
    """
    plt.clf()  
    plt.figure(figsize=(12, 8)) 
    shap.plots.beeswarm(shap_values, max_display=20) 
    plt.xticks(fontsize=10)
    plt.tight_layout()
    plt.show()


def display_description(key, descriptions):
    """
    Display the description for a given SHAP plot using its key.

    Args:
        key (str): The key for the SHAP plot in the descriptions JSON.
        descriptions (dict): The dictionary containing the descriptions.
    """
    
    st.write(descriptions[key]["description"])
    st.write(descriptions[key]["where_it_helps"])
    st.write(descriptions[key]["how_to_use"])
    st.write(descriptions[key]["requirements"])


@st.cache_data
def load_descriptions():
    with open("xai_banking/utils/descriptions.json", "r") as f:
        descriptions = json.load(f)
    return descriptions


def create_lime_vs_shap_info():
# Expander: When to Use LIME or SHAP
    with st.expander("When to Use LIME or SHAP?"):
        st.write("""
        **LIME (Local Interpretable Model-Agnostic Explanations):**
        - **When to Use**:
            - Use LIME when you need quick and local explanations for individual predictions.
            - Works well with any model, as it does not require access to model internals (black-box compatible).
            - Useful for debugging a specific instance or for models that are not inherently explainable.
        - **Strengths**:
            - Simple to use and interpret for small datasets.
            - Offers model-agnostic explanations, meaning it works with any machine learning model.
        - **Limitations**:
            - Explanations can vary depending on sampling.
            - It provides local explanations, but not a global understanding of the model.

        **SHAP (SHapley Additive exPlanations):**
        - **When to Use**:
            - Use SHAP when you need both local and global explanations for your model.
            - Ideal for understanding feature importance across the dataset and for individual predictions.
            - Works well with models that provide access to feature contributions (e.g., tree-based models, deep learning models).
        - **Strengths**:
            - Provides consistent and theoretically sound explanations based on Shapley values.
            - Suitable for explaining predictions globally (across the entire dataset) and locally (specific data points).
            - Includes a variety of visualization tools (Summary Plot, Beeswarm Plot, etc.).
        - **Limitations**:
            - Computationally expensive for large datasets or complex models.
            - May require more time and resources compared to LIME.

        **Key Differences:**
        - LIME explains individual predictions by approximating the model locally, while SHAP is based on cooperative game theory and assigns contributions to features globally and locally.
        - SHAP provides more comprehensive insights (e.g., feature importance, interactions) at the cost of higher computational complexity.
        - LIME is faster and easier to use for small datasets and models, making it a better choice for quick insights or debugging specific predictions.

        **Choosing the Right Tool:**
        - Use **LIME** if:
            - Your dataset is small, and you need a quick explanation for individual predictions.
            - You are working with a black-box model with no access to internal computations.
        - Use **SHAP** if:
            - You need both global and local explanations for your model.
            - You want a deeper understanding of how your model behaves across the entire dataset.
            - You have the computational resources to handle more complex explainability tasks.
        """)
