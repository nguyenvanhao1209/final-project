import streamlit as st
from authentication import Auth
import streamlit_shadcn_ui as ui
from google_login import google_login

def login():
    st.subheader("Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Login"):
            try:
                user = Auth.sign_in(email, password)
                st.switch_page('pages/home.py')
            except Exception as e:
                st.error(f"Error: {e}")
    with col2:
        if st.button("Sign Up"):
            st.switch_page('pages/signup.py')
    with col3:
        google_login()

def main():
    login()

if __name__ == '__main__':
    main()