import streamlit as st
import streamlit.components.v1 as components
import base64

logo_path = r'logo1.png'
with open(logo_path, "rb") as image_file:
    encoded_logo = base64.b64encode(image_file.read()).decode('utf-8')

st.set_page_config(
    page_title="Streamlit App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

st.markdown(f"""
    <nav class="navbar">
        <a class="navbar-brand" href="#" id="nav-a">
            <img src="data:image/png;base64,{encoded_logo}" alt="logo" />
            <span>YOUR SAAS</span>
        </a>
        <a href="/login" target="_self" class="login-button">Login</a>
        <a href="/signup" target="_self" class="signup-button">Sign Up</a>
    </nav>
    """, unsafe_allow_html=True)

st.title("Welcome to Streamlit App")

