import streamlit as st
from authentication import Auth
from google_login import get_logged_in_user_email

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

st.markdown(f"""
    <nav class="navbar">
        <a class="navbar-brand" href="#" id="nav-a">
            <span>YOUR SAAS</span>
        </a>
    </nav>
    """, unsafe_allow_html=True)

def get_current_login():
    auth_instance = get_logged_in_user_email()
    st.write(f"hello {auth_instance.LoginUser()}")

    if st.button('Logout'):
        logout()

def logout():
    Auth.logout()
    st.switch_page('pages/login.py')


def main():
    get_current_login()

if __name__ == "__main__":
    main()