import streamlit as st
from authentication import Auth

def register():
    st.subheader("Register")

    email = st.text_input("Email")
    name = st.text_input("Name")
    password = st.text_input("Password", type='password')
    repassword = st.text_input("Re-enter Password", type='password')


    if st.button("Sign Up"):
        if password != repassword:
            st.error("Passwords do not match")
        else:
            try:
                user = Auth.register(email, password, name)
                st.switch_page('pages/login.py')
            except Exception as e:
                st.error(f"Error: {e}")
    
    if st.button("Login"):
        st.switch_page('pages/login.py')

def main():
    register()

if __name__ == '__main__':
    main()