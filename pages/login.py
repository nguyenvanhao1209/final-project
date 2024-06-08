import streamlit as st
from authentication import Auth
import streamlit_shadcn_ui as ui
from services.google_login import google_login
from local_components import card_container
from PIL import Image



im = Image.open("image/logo-1.png")

st.set_page_config(
    page_title="Login | Kogga",
    page_icon=im,
    layout="wide",
    initial_sidebar_state="expanded",
)

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
    

    


def login():
    
    image = Image.open("image/logo-no-background.png")
    col1_logo, col2_logo, col3_logo = st.columns(3)
    with col2_logo:
        col1_logo_in, col2_logo_in, col3_logo_in = st.columns(3)
        with col2_logo_in:
            new_image = image.resize((150, 75))
            st.image(new_image)
    
    col1b, col2b, col3b = st.columns(3)
    with col2b:
        st.markdown(
                    """
                    <style>
                    .c {
                        margin-top: 10px ;
                        }
                    </style>
    
                    <div class="c"></div>
                    """,
                    unsafe_allow_html=True
                )
        
        with card_container(key="login-form"):
            
            st.markdown("## Welcome!")
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
                    except Exception:
                        st.toast("Someting went wrong", icon='🚨')
            with col2:
                if st.button("Sign Up", key='signup-btn', type="primary", use_container_width=True):
                    st.switch_page('pages/signup.py')

            google_login()
             

def main():
    login()

if __name__ == '__main__':
    main()