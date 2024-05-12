import streamlit as st
from authentication import Auth
from services.google_login import get_logged_in_user_email
from services.post import create_post
import datetime

def main():
    auth_instance = get_logged_in_user_email()
    st.write(f"hello {auth_instance.LoginUser()}")
    title = st.text_input("Title")
    content = st.text_area("Content")
    files = st.file_uploader("Upload files", accept_multiple_files=True)
    image = st.file_uploader("upload a image")

    if st.button("Submit"):
        create_post(title, content, auth_instance.LoginUser(), datetime.datetime.now(), files, image)

if __name__ == "__main__":
    main()
