import streamlit as st

def load_global_css(css_file_path="assets/style.css"):
    """Reads and injects custom CSS into the Streamlit app."""
    try:
        with open(css_file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file not found at {css_file_path}")