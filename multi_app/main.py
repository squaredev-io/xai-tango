import streamlit as st
import os
import importlib
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# App selection list
apps = {
    "Banking Pilot": "banking_pilot",
    "Fmake": "fmake"
}

# Sidebar menu
st.sidebar.title("App Selector")
selected_app = st.sidebar.selectbox("Choose an app", options=list(apps.keys()))

# Load and run the selected app
if selected_app:
    st.sidebar.success(f"Running: {selected_app}")

    # Dynamically import and execute the selected app
    app_module = importlib.import_module(apps[selected_app])
    app_module.main()
