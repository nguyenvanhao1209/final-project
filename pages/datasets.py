import streamlit as st
from authentication import Auth
from services.google_login import get_logged_in_user_email
from services.post import list_post
from PIL import Image
import requests
from io import BytesIO
from utils import time_difference, format_file_size, get_file_extension

def main():
    auth_instance = get_logged_in_user_email()
    posts = list_post()
    # Create a list to hold the columns
    cols = [st.columns(4) for _ in range(len(posts) // 4 + (len(posts) % 4 > 0))]

    for i, post in enumerate(posts):
        # Calculate the row and column index
        row = i // 4
        col = i % 4

        # Fetch the image from the URL
        response = requests.get(post.image)
        try:
            image = Image.open(BytesIO(response.content))
        except Exception:
            st.write("Unable to load image")

        # Display the post in a card
        with cols[row][col]:
            st.image(image, use_column_width=True)
            st.write(f"Title: {post.title}")
            st.write(f"Author: {post.author.name}")
            st.write(f"Date: {time_difference(post.datetime.strftime('%Y-%m-%d %H:%M:%S'))}")
            file_sizes = 0
            file_extensions = set()
            for file in post.files:
            # Get the file size from the 'Content-Length' header
                file_response = requests.head(file)
                file_size = file_response.headers.get('Content-Length', 'Unknown')
                file_sizes += int(file_size)
                file_extensions.add(get_file_extension(file))

            st.write(f"{file_extensions}")
            st.write(f"File size: {format_file_size(file_sizes)}")

            if len(post.files) == 1:
                st.write(f"1 File")
            else:
                st.write(f"{len(post.files)} Files")

            if st.button(f"Detail {post.title}"):
                st.session_state.current_post = post
                st.switch_page("pages/detail_dataset.py")


    if st.button('Create Post'):
        st.switch_page("pages/create_dataset.py")

if __name__ == "__main__":
    main()
