import streamlit as st

def create_additional_css():
    logo_url = "https://tango-project.eu/themes/custom/tango01/images/logo-tango.svg"

    # Display the logo in the app
    st.logo(logo_url, link="https://tango-project.eu")

    custom_css = """
    <style>
    /* Main app background gradient */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to right, #0575E6, #021B79);
        color: white;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #021B79, #0575E6);
        color: white;
        border-right: 2px solid #fff;
    }

    /* Sidebar content text color */
    [data-testid="stSidebar"] .css-1d391kg {
        color: white;
    }

    /* Title and header styling */
    h1, h2, h3 {
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }

    /* Button styling */
    button {
        background: linear-gradient(to right, #0575E6, #021B79);
        color: white !important;
        border: none;
        border-radius: 5px;
        font-weight: bold;
        font-size: 16px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    /* Input boxes */
    input, textarea {
        background-color: #f0f8ff;
        border: 1px solid #021B79;
        border-radius: 5px;
        color: #021B79;
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