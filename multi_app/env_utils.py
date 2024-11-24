from dotenv import load_dotenv
import os
import streamlit as st

@st.cache_data
def load_env_vars():
    load_dotenv()
    BANKING_MODEL_PATH = os.getenv("BANKING_MODEL_PATH")
    BANKING_DATA_PATH = os.getenv("BANKING_DATA_PATH")
    VISION_MODEL_PATH = os.getenv("VISION_MODEL_PATH")
    VISION_DATA_PATH = os.getenv("VISION_DATA_PATH")

    return BANKING_MODEL_PATH, BANKING_DATA_PATH, VISION_MODEL_PATH, VISION_DATA_PATH