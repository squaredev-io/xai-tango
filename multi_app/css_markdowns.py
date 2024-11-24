import streamlit as st

def create_additional_css():
    logo_url = "https://tango-project.eu/themes/custom/tango01/images/logo-tango.svg"

    # Display the logo in the app
    st.logo(logo_url, link="https://tango-project.eu")

    custom_css = """
    <style>
    /* Main app background gradient */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle, #1E5E9C, #0E3765); /* Higher contrast radial gradient */
        color: white;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: radial-gradient(circle, #0E3765, #1E5E9C); /* Match tones with darker edges */
        color: white;
        border-right: 2px solid #ffffff;
    }

    /* Sidebar content text color */
    [data-testid="stSidebar"] .css-1d391kg {
        color: white;
    }

    /* Title and header styling */
    h1, h2, h3 {
        color: #ffffff;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.6); /* Slightly stronger shadow for readability */
    }

    /* Button styling */
    button {
        background: linear-gradient(to right, #1E5E9C, #0E3765); /* Consistent with darker gradient */
        color: white !important;
        border: none;
        border-radius: 5px;
        font-weight: bold;
        font-size: 16px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3); /* Slightly stronger hover effect */
    }

    /* Input boxes */
    input, textarea {
        background-color: #e1effc; /* Lightened for contrast with dark background */
        border: 1px solid #0E3765;
        border-radius: 5px;
        color: #0E3765;
        font-weight: bold;
    }

    /* Streamlit progress bar and other default widgets */
    .css-17eq0hr {
        color: white;
    }

    footer {
        color: white;
    }

    /* Remove Streamlit's default footer and branding */
    footer:after {
        content: '';
    }
    </style>
    """





    # Inject the CSS into the Streamlit app
    st.markdown(custom_css, unsafe_allow_html=True)

    # Footer content
    footer = """
    <div style="
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #0D47A1;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 12px;
        z-index: 1000;">
        <img src="https://tango-project.eu/sites/default/files/europe-flag.png" 
            alt="EU Flag" 
            style="width: 50px; vertical-align: middle; margin-right: 10px;">
        <span>This project has received funding from the European Union’s HE research and innovation programme under the grant agreement No. 101070052</span>
    </div>
    """

    # Inject footer HTML
    st.markdown(footer, unsafe_allow_html=True)

    sidebar_footer = """
    <div class="fixed-sidebar-footer" style="
        position: fixed;
        left: 0;
        bottom: 0;
        width: 21rem;  /* Matches Streamlit's sidebar width */
        padding: 1rem;
        text-align: center;
        background-color: transparent;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
        z-index: 999;
    ">
        <img src="https://squaredev.io/wp-content/uploads/2022/11/Squaredev-black-B-trans.png" 
            alt="Squaredev Logo" 
            style="width: 80px; margin-bottom: 5px;">
        <p style="margin: 0; color: rgba(250, 250, 250, 0.8); font-size: 0.8rem;">Powered by Squaredev</p>
    </div>

    <style>
        /* Ensure sidebar content doesn't overlap with footer */
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            padding-bottom: 5rem;
        }
        
        /* Adjust for different screen sizes */
        @media (max-width: 768px) {
            .fixed-sidebar-footer {
                width: 100%;
            }
        }
    </style>
    """
    # Inject sidebar footer
    st.sidebar.markdown(sidebar_footer, unsafe_allow_html=True)

import streamlit as st

def display_xai_library_info():
    """
    Displays an overview of the Explainable AI (XAI) library, its techniques, applications,
    and visualizations in a Streamlit app.
    """
    xai_info = {
        "overview": (
            "A library of explainable AI (XAI) techniques is developed to enhance transparency and trust "
            "in AI systems by offering insights into the reasoning behind model predictions and decisions."
        ),
        "vision_models": {
            "description": (
                "For vision models, techniques like SHAP, GradCAM, and Convolutional Layer Activation Analysis "
                "are integrated to uncover pixel-level contributions and highlight the most influential regions in images."
            ),
            "use_cases": [
                "Image classification",
                "Object detection",
                "Pixel-level feature contribution analysis"
            ],
            "techniques": ["SHAP", "GradCAM", "Convolutional Layer Activation Analysis"]
        },
        "tabular_models": {
            "description": (
                "For tabular models, tools like LIME, SHAP, and feature contribution analysis are applied to "
                "identify and explain the impact of specific features on predictions."
            ),
            "use_cases": [
                "Risk classification",
                "Credit scoring",
                "Feature-level analysis of predictions"
            ],
            "techniques": ["LIME", "SHAP", "Feature Contribution Analysis"]
        },
        "global_vs_local": {
            "description": (
                "The framework supports both global and local explainability. Global methods provide an overarching "
                "view of feature importance across datasets, while local approaches offer granular explanations for individual predictions."
            )
        },
        "visualization": {
            "description": (
                "Visualization plays a key role in this library. Outputs such as SHAP summary plots, waterfall plots, "
                "and activation heatmaps allow users to intuitively interpret the results."
            ),
            "techniques": ["SHAP summary plots", "Waterfall plots", "Activation heatmaps"]
        },
        "impact": (
            "Through these efforts, privacy and trust concerns are addressed, while gaps in the design or deployment "
            "of AI algorithms are identified and rectified."
        )
    }

    # Display XAI information
    st.title("Explainable AI Library Overview")
    st.write(xai_info["overview"])
    
    st.subheader("Vision Models")
    st.write(xai_info["vision_models"]["description"])
    st.write("**Use Cases:**", ", ".join(xai_info["vision_models"]["use_cases"]))
    st.write("**Techniques:**", ", ".join(xai_info["vision_models"]["techniques"]))

    st.subheader("Tabular Models")
    st.write(xai_info["tabular_models"]["description"])
    st.write("**Use Cases:**", ", ".join(xai_info["tabular_models"]["use_cases"]))
    st.write("**Techniques:**", ", ".join(xai_info["tabular_models"]["techniques"]))

    st.subheader("Global vs. Local Explainability")
    st.write(xai_info["global_vs_local"]["description"])

    st.subheader("Visualization Techniques")
    st.write(xai_info["visualization"]["description"])
    st.write("**Techniques:**", ", ".join(xai_info["visualization"]["techniques"]))

    st.info(xai_info["impact"])
