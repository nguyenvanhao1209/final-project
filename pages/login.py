import streamlit as st
from authentication import Auth
import streamlit_shadcn_ui as ui
from services.google_login import google_login
from local_components import card_container

import json 
import requests
from streamlit_lottie import st_lottie 


st.set_page_config(
    page_title="Streamlit App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
    

    


def login():
    col1b, col2b, col3b = st.columns([1,2,3])
    with col2b:
        with card_container(key="login-form"):
            st.markdown("#### Login to my app")
            col1, col2 = st.columns([100,1])
            
            with col1:
                email = st.text_input("Email", placeholder="Input your email")
                password = st.text_input("Password", type="password", placeholder="Input your password")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Login", key='login-btn', type="primary", use_container_width=True):
                    try:
                        user = Auth.sign_in(email, password)
                        st.switch_page('pages/home.py')
                    except Exception as e:
                        st.error(f"Error: {e}")
            with col2:
                if st.button("Sign Up", key='signup-btn', type="primary", use_container_width=True):
                    st.switch_page('pages/signup.py')
            
            google_login()
            
            
    with col3b:
        url = requests.get( 
        "https://lottie.host/97faaffa-7b41-495e-887e-1319f6f89d6d/vaa65SdYvp.json") 
        url_json = dict() 
        if url.status_code == 200: 
            url_json = url.json() 
        else: 
            print("Error in URL") 



        st_lottie(url_json, reverse=True, height=600, width=600, speed=1, loop=True, quality='high', key='Car' ) 

def main():
    login()

if __name__ == '__main__':
    main()