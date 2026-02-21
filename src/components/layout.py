import streamlit as st

def set_enterprise_layout():
    """Sets consistent page branding across the framework."""
    st.set_page_config(
        page_title="HFC | Championship Blueprint",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )