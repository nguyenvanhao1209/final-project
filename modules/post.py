import streamlit as st
from services.comment import create_comment, list_comment
from datetime import datetime
from local_components import card_container
from services.post import list_post
import requests
from io import BytesIO
from utils import time_difference, format_file_size, get_file_extension
import zipfile
import io
from utils import get_file
from PIL import Image
import pandas as pd
from services.google_login import get_logged_in_user_email

class Post:
    def all_post():
        st.write(" # Tập dữ liệu # ")
        col1, col2 = st.columns([5,1])
        with col1:
            st.write("Khám phá, phân tích và chia sẻ dữ liệu chất lượng")
        with col2:
            if st.button('Upload your data', type='primary', key='upload_data'):
                st.switch_page("pages/create_dataset.py")
            
        st.markdown("---")
        posts = list_post()
        # Create a list to hold the columns
        cols = [st.columns(3) for _ in range(len(posts) // 3 + (len(posts) % 3 > 0))]

        for i, post in enumerate(posts):
            # Calculate the row and column index
            row = i // 3
            col = i % 3

            # Fetch the image from the URL
            response = requests.get(post.image)
            try:
                image = Image.open(BytesIO(response.content))
                
            except Exception:
                st.write("Unable to load image")

            # Display the post in a card
            with cols[row][col]:
                with card_container(key="dataset-card"):
                    resized_image = image.resize((300,150), Image.LANCZOS)
                    file_sizes = 0
                    file_extensions = set()
                    for file in post.files:
                    # Get the file size from the 'Content-Length' header
                        file_response = requests.head(file)
                        file_size = file_response.headers.get('Content-Length', 'Unknown')
                        file_sizes += int(file_size)
                        file_extensions.add(get_file_extension(file))

                    
                    
                    st.image(resized_image, use_column_width=True)
                    st.write(f"Title: {post.title}")
                    st.write(f"{post.id}")
                    st.write(f"Author: {post.author.name}")
                    col1, col2 = st.columns(2)
                    with col1:
                        if len(post.files) == 1:
                            st.write(f"1 File")
                        else:
                            st.write(f"{len(post.files)} Files")
                            
                        st.write(f"File size: {format_file_size(file_sizes)}")
                        
                    with col2:
                        st.write(f"Date: {time_difference(post.datetime.strftime('%Y-%m-%d %H:%M:%S'))}")
                        st.write(f"Type: {file_extensions}")
                        
                    if st.button(f"Detail {post.title}", type='primary'):
                        Post.detail_post(post)

    @st.experimental_dialog("Detail post", width="large")
    def detail_post(post):
        auth_instance = get_logged_in_user_email()
        st.write(f"detail {post.title}")
        for file in post.files:
            df = pd.read_csv(file)
            st.dataframe(df)

        new_comment = st.text_input('Write a comment')

        if st.button('Submit Comment'):
            create_comment(new_comment, auth_instance.LoginUser(), datetime.now(), post)

        comments = list_comment(post.id)

        for comment in comments:
            st.write(f"{comment.content}")