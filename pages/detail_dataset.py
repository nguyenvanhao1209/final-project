import streamlit as st
import requests
import zipfile
import io
from utils import get_file
import pandas as pd
from services.comment import create_comment, list_comment
from datetime import datetime
from services.google_login import get_logged_in_user_email

def download_files_as_zip(post):
    # Create a BytesIO object to store the zip file in memory
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file in post.files:
            file_response = requests.get(file)
            file_name = get_file(file)
            zip_file.writestr(file_name, file_response.content)

    zip_buffer.seek(0)

    st.download_button(
        label="Download All Files",
        data=zip_buffer,
        file_name=f"{post.title}.zip",
        mime="application/zip"
    )

def get_column_info():
    pass

def main():
    auth_instance = get_logged_in_user_email()
    st.write(f"hello {auth_instance.LoginUser()}")
    post = st.session_state.current_post
    st._set_query_params(dataset=post.title)
    st.write(f"Title: {post.title}")
    st.write(f"Author: {post.author.name}")

    download_files_as_zip(post)

    st.header("About Dataset")

    st.write(post.content)

    for file in post.files:
        if st.button(f"{get_file(file)}"):
            dataframe = pd.read_csv(file)
            st.dataframe(dataframe)

    new_comment = st.text_input('Write a comment')

    if st.button('Submit Comment'):
        create_comment(new_comment, auth_instance.LoginUser(), datetime.now(), post)

    # Get and display comments
    comments = list_comment(post)
    for comment in comments:
        st.write(f"{comment['author']['name']}: {comment['content']}")




if __name__ == "__main__":
    main()