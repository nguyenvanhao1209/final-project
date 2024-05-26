import streamlit as st
from services.comment import create_comment, list_comment
from datetime import datetime
from local_components import card_container
from services.post import list_post, create_post, update_post, delete_post, search_post_by_title, search_post_by_file_type, search_post_by_file_size
import requests
from io import BytesIO
from utils import time_difference, format_file_size, get_file_extension, calculate_file_size
import zipfile
import io
from utils import get_file
from PIL import Image
import pandas as pd
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

@st.experimental_dialog("Filter box", width="small")
def filter_option():
    st.write("File types")
    filter_col = ['csv', 'json', 'sql', 'xlsx']
    selected_file_types = st.multiselect("File types", filter_col, label_visibility="collapsed")
    st.session_state.selected_file_types = selected_file_types
    st.write("File size")
    colnb1, cols1, colnb2, cols2 = st.columns(4)
    with colnb1:
        min_size = st.number_input("Min", placeholder="Min", label_visibility="collapsed")
    with cols1:
        size_type_min = st.selectbox("Min size type", ['kB', 'MB', 'GB'], label_visibility="collapsed")
    with colnb2:
        max_size = st.number_input("Max", placeholder="Max", label_visibility="collapsed")
    with cols2:
        size_type_max = st.selectbox("Max size type", ['kB', 'MB', 'GB'], label_visibility="collapsed")
    if calculate_file_size(min_size, size_type_min) > calculate_file_size(max_size, size_type_max):
        st.warning("Invalid file size range entered")
    else:
        if st.button("Apply"):
            min_file_size = calculate_file_size(min_size, size_type_min)
            max_file_size = calculate_file_size(max_size, size_type_max)
            st.write(selected_file_types, min_file_size, max_file_size)

class Post:
    def all_post():
        auth_instance = get_logged_in_user_email()
        st.write(" # Tập dữ liệu # ")
        col1, col2 = st.columns([5, 1])
        with col1:
            st.write("Khám phá, phân tích và chia sẻ dữ liệu chất lượng")
        with col2:
            if st.button("Upload your data", type="primary", key="upload_data"):
                Post.create_post()

        colf1, colf2 = st.columns([7,1])
        with colf1:
            search_text = st.text_input(label="search", placeholder="Search some thing...", label_visibility="hidden")
        with colf2:
            with st.popover("Filters", use_container_width=False):
                st.write("File types")
                filter_col = ['csv', 'json', 'sql', 'xlsx']
                selected_file_types = st.multiselect("File types", filter_col, label_visibility="collapsed")
                st.write("File size")
                colnb1, cols1, colnb2, cols2 = st.columns(4)
                with colnb1:
                    min_size = st.number_input("Min", placeholder="Min", label_visibility="collapsed")
                with cols1:
                    size_type_min = st.selectbox("Min size type", ['kB', 'MB', 'GB'], label_visibility="collapsed")
                with colnb2:
                    max_size = st.number_input("Max", placeholder="Max", label_visibility="collapsed")
                with cols2:
                    size_type_max = st.selectbox("Max size type", ['kB', 'MB', 'GB'], label_visibility="collapsed")
                selected_file_types_final = []
                min_file_size = None
                max_file_size = None
                if calculate_file_size(min_size, size_type_min) > calculate_file_size(max_size, size_type_max):
                    st.warning("Invalid file size range entered")
                else:
                    colvui, colap, colcl = st.columns([5,1,1])
                    with colap:
                        if st.button("Apply"):
                            selected_file_types_final = selected_file_types
                            min_file_size = calculate_file_size(min_size, size_type_min)
                            max_file_size = calculate_file_size(max_size, size_type_max)
                    with colcl:
                        if st.button("Clear"):
                            selected_file_types_final = []
                            min_file_size = None
                            max_file_size = None

        st.markdown("---")
        if search_text == "":
            posts = list_post()
        else:
            posts = search_post_by_title(search_text)

        if selected_file_types_final:
            posts = search_post_by_file_type(posts, selected_file_types_final)

        if min_file_size is not None and max_file_size is not None:
            posts = search_post_by_file_size(posts, min_file_size, max_file_size)


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
                    resized_image = image.resize((300, 100), Image.LANCZOS)
                    file_sizes = 0
                    file_extensions = set()
                    for file in post.files:
                        # Get the file size from the 'Content-Length' header
                        file_response = requests.head(file)
                        file_size = file_response.headers.get(
                            "Content-Length", "Unknown"
                        )
                        file_sizes += int(file_size)
                        file_extensions.add(get_file_extension(file))

                    st.image(resized_image, use_column_width=True)
                    st.write(f"Title: {post.title}")
                    st.write(f"Author: {post.author.name}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if len(post.files) == 1:
                            st.write(f"1 File")
                        else:
                            st.write(f"{len(post.files)} Files")

                    with col2:
                        st.write(f"{format_file_size(file_sizes)}")

                    with col3:
                        st.write(f"{file_extensions}")

                    st.write(
                        f"Time: {time_difference(post.datetime.strftime('%Y-%m-%d %H:%M:%S'))}"
                    )
                    
                    btncol1, btncol2, btncol3 = st.columns(3)
                    with btncol1:
                        if st.button(f"Detail", type="primary", key=f"Detail {post.title}"):
                            Post.detail_post(post)
                    with btncol2:
                        if post.author.name == auth_instance.LoginUser().name:
                            if st.button(f"Update", type="primary", key=f"Update {post.title}"):
                                Post.update_post(post)
                    with btncol3:
                        if post.author.name == auth_instance.LoginUser().name:
                            if st.button(f"Delete", type="primary", key=f"Delete {post.title}"):
                                Post.delete_post(post)

    @st.experimental_dialog("Detail post", width="large")
    def detail_post(post):
        download_files_as_zip(post)
        auth_instance = get_logged_in_user_email()
        st.write(f"detail {post.title}")
        for file in post.files:
            if get_file_extension(file) == "csv":
                df = pd.read_csv(file)
            elif get_file_extension(file) == "xlsx":
                df = pd.read_excel(file)
            elif get_file_extension(file) == "json":
                df = pd.read_json(file)
            elif get_file_extension(file) == "sql":
                df = pd.read_sql(file)
            else:
                df = None
            st.dataframe(df)

        new_comment = st.text_input('Write a comment')

        if st.button('Submit Comment'):
            create_comment(new_comment, auth_instance.LoginUser(), datetime.now(), post)

        comments = list_comment(post.id)

        for comment in comments:
            st.write(f"{comment.content}")

    @st.experimental_dialog("Create post", width="large")
    def create_post():
        st.write("Share your own dataset")
        auth_instance = get_logged_in_user_email()
        with st.form("upload_form"):
            title = st.text_input("Title", max_chars=100)
            content = st.text_area("Content")
            files = st.file_uploader("Upload files", accept_multiple_files=True)
            image = st.file_uploader("Upload a image")
            submitted = st.form_submit_button("Submit", type="primary")
            if submitted:
                create_post(
                    title,
                    content,
                    auth_instance.LoginUser(),
                    datetime.now(),
                    files,
                    image,
                )
                st.rerun()

    @st.experimental_dialog("Update post", width="large")
    def update_post(post):
        st.write("Update your dataset")
        auth_instance = get_logged_in_user_email()
        with st.form("update_form"):
            title = st.text_input("Title", max_chars=100, value=f"{post.title}")
            content = st.text_area("Content", value=f"{post.content}")
            files = st.file_uploader("Upload files", accept_multiple_files=True)
            image = st.file_uploader("Upload a image")
            submitted = st.form_submit_button("Submit", type="primary")
            if submitted:
                if files == []:
                    files = post.files
                if image is None:
                    image = post.image
                update_post(
                    post.id,
                    title,
                    content,
                    auth_instance.LoginUser(),
                    datetime.now(),
                    files,
                    image,
                )
                st.rerun()

    def delete_post(post):
        delete_post(post_id=post.id)
        st.rerun()
