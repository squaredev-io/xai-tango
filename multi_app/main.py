import streamlit as st
import os
import importlib
import sys
from dotenv import load_dotenv
from css_markdowns import create_additional_css, display_xai_library_info

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
load_dotenv()

# Fetch the valid token from the .env file
VALID_TOKEN = os.getenv("STREAMLIT_TOKEN")

if not VALID_TOKEN:
    raise ValueError("STREAMLIT_TOKEN is not set in the .env file.")

# Initialize session state for authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Token input form
if not st.session_state.authenticated:
    with st.sidebar.form("token_form"):
        st.write("🔒 Enter the token to unlock:")
        token_input = st.text_input("Token", type="password")
        submit_button = st.form_submit_button("Submit")

        if submit_button:
            if token_input == VALID_TOKEN:
                st.session_state.authenticated = True
                st.sidebar.success("Token verified. App unlocked!")
                st.rerun()
            else:
                st.sidebar.error("Invalid token. Please try again.")

# If authenticated, show app functionality
if st.session_state.authenticated:
    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

    # Sidebar menu
    st.sidebar.title("Pilot Selector")
    # App selection list
    apps = {
        "Banking Pilot": "banking_pilot",
        "Fmake": "fmake"
    }
    
    selected_app = st.sidebar.selectbox("Choose an app", options=list(apps.keys()))

    # Load and run the selected app
    if selected_app:
        st.sidebar.success(f"Running: {selected_app}")

        # Dynamically import and execute the selected app
        app_module = importlib.import_module(apps[selected_app])
        app_module.main()

else:
    st.sidebar.warning("🔒 Please enter a valid token in the sidebar to continue.")
    display_xai_library_info()

# Add additional CSS
create_additional_css()
