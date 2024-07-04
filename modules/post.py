import streamlit as st
from services.comment import create_comment, list_comment, delete_comment
from datetime import datetime
from local_components import card_container
from services.post import list_post, create_post, update_post, delete_post, search_post_by_title, search_post_by_file_type, search_post_by_file_size, update_downloaded, get_user_posts
from pygwalker.api.streamlit import StreamlitRenderer
import requests
from io import BytesIO
from utils import time_difference, format_file_size, get_file_extension, calculate_file_size, display_vote_detail
import zipfile
import io
from utils import get_file
from PIL import Image
import pandas as pd
from services.google_login import get_logged_in_user_email
from services.vote import create_vote, get_point_vote, is_voted, delete_vote, change_vote, get_vote_user, get_vote_counts, get_total_votes
import time
from streamlit_star_rating import st_star_rating
import streamlit_shadcn_ui as ui

@st.experimental_fragment
def download_files_as_zip(post):
    # Initialize the flag in the session state
    if f'has_updated_{post.id}' not in st.session_state:
        st.session_state[f'has_updated_{post.id}'] = False

    # Create a BytesIO object to store the zip file in memory
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file in post.files:
            file_response = requests.get(file)
            file_name = get_file(file)
            zip_file.writestr(file_name, file_response.content)

    zip_buffer.seek(0)

    if st.download_button(
        label=f"Download All Files",
        data=zip_buffer,
        file_name=f"{post.title}.zip",
        mime="application/zip",
        type="primary",
    ):
        if not st.session_state[f'has_updated_{post.id}']:
            # Update the flag in the session state
            st.session_state[f'has_updated_{post.id}'] = True
            update_downloaded(post.id)

@st.experimental_fragment
def handle_comment(post, auth_instance):
    if is_voted(auth_instance.LoginUser().id, post.id) == False:
        placeholder = st.empty()
        container = placeholder.container(height=200, border=False)
        with container:
            st.markdown("""<h1 style="position: relative; top: -15px;">Create your rating</h1>""", unsafe_allow_html=True)
            vote_point = st_star_rating("", maxValue=5,defaultValue=0,size=30,customCSS = "div {padding-left:100px; height: 60px;}")
            new_comment = st.chat_input("Write a comment")
        if new_comment and vote_point:
            create_vote(auth_instance.LoginUser(), post, vote_point)
            create_comment(new_comment, auth_instance.LoginUser(), datetime.now(), post)
            placeholder.empty()
    else:
        pass
    
    if get_total_votes(post.id):
        st.markdown("""<h1 style="position: relative; top: -20px;">Ratings and comments</h1>""", unsafe_allow_html=True)
        display_vote_detail(get_vote_counts(post.id), get_point_vote(post.id), get_total_votes(post.id))

    comments = list_comment(post.id)

    for comment in comments:
        with st.container(height=150, border=True):
            st.write(f"## {comment.user.name}")
            col1, col2 = st.columns([1,3])
            with col1:
                st_star_rating("", maxValue=5, defaultValue=get_vote_user(comment.user.id, post.id), size=20, read_only=True, key=f"vote{comment.user.id}{post.id}")
            with col2:
                st.markdown(f"""<div style="margin-top:5px;">{time_difference(comment.datetime.strftime('%Y-%m-%d %H:%M:%S'))}</div>""", unsafe_allow_html=True)
            st.write(f"{comment.content}")
class Post:
    def all_post():
        auth_instance = get_logged_in_user_email()
        st.write(" # Datasets # ")
        col1, col2 = st.columns([5, 1])
        with col1:
            st.write("Discover, analyze and share quality data")
        with col2:
            if st.button("Upload your data", type="primary", key="upload_data"):
                Post.create_post()

        colf1, colf2 = st.columns([7,1])
        with colf1:
            search_text = st.text_input(label="search", placeholder="Search some thing...", label_visibility="collapsed")
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
                min_file_size = 0
                max_file_size = 0
                my_data = False
                if calculate_file_size(min_size, size_type_min) > calculate_file_size(max_size, size_type_max):
                    st.warning("Invalid file size range entered")
                else:
                    colvui, colap, colcl = st.columns([5,1,1])
                    with colvui:
                        my_data = st.toggle("My posts", value=False, key="get_my_posts")
                    with colap:
                        if st.button("Apply"):
                            selected_file_types_final = selected_file_types
                            min_file_size = calculate_file_size(min_size, size_type_min)
                            max_file_size = calculate_file_size(max_size, size_type_max)
                    with colcl:
                        if st.button("Clear"):
                            selected_file_types_final = []
                            min_file_size = 0
                            max_file_size = 0

        st.markdown("---")
        if search_text == "":
            posts = list_post()
        else:
            posts = search_post_by_title(search_text)

        if selected_file_types_final:
            posts = search_post_by_file_type(posts, selected_file_types_final)

        if min_file_size != 0 or max_file_size != 0:
            posts = search_post_by_file_size(posts, min_file_size, max_file_size)

        if my_data:
            posts = get_user_posts(posts, auth_instance.LoginUser().id)


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
                        file_size = file_response.headers.get("Content-Length", "Unknown")
                        file_sizes += int(file_size)
                        file_extensions.add(get_file_extension(file))

                    st.image(resized_image, use_column_width=True)
                    st.write(f"###### {post.title} - <u> {post.author.name} </u>", unsafe_allow_html=True)
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

                    col1, col2, col3 = st.columns([3,1,1])
                    with col1:
                        st.markdown(f"<small>Updated {time_difference(post.datetime.strftime('%Y-%m-%d %H:%M:%S'))}</small>", unsafe_allow_html=True)
                    with col2:
                        if get_point_vote(post.id) != 0.0:
                            st.write(f"{get_point_vote(post.id)}⭐")
                        else:
                            pass
                    with col3:
                        if post.downloaded != 0:
                            st.write(f"{post.downloaded}⬇️")
                        else:
                            pass

                    btncol1, btncol2, btncol3 = st.columns(3)
                    with btncol1:
                        if ui.button("🔍", key=f"Detail {post.id}", class_name="text-white bg-slate-200 font-bold w-10 h-10 py-2 px-2 rounded-full"):
                            Post.detail_post(post)
                    with btncol2:
                        if post.author.name == auth_instance.LoginUser().name:
                            if ui.button("🔄", key=f"Update {post.id}", class_name="text-white bg-slate-200 font-bold w-10 h-10 py-2 px-2 rounded-full"):
                                Post.update_post(post)
                    with btncol3:
                        if post.author.name == auth_instance.LoginUser().name:
                            if ui.button("✖️", key=f"Delete {post.id}", class_name="text-white bg-slate-200 font-bold w-10 h-10 py-2 px-2 rounded-full"):
                                Post.delete_post(post)

    @st.experimental_dialog("Detail post", width="large")
    def detail_post(post):
        auth_instance = get_logged_in_user_email()
        col1d, col2d = st.columns([3, 1])
        with col1d:
            st.write(f"### Detail {post.title}")
        with col2d:
            download_files_as_zip(post)

        with st.container(border=True):
            st.write("#### About dataset")
            st.write(f"{post.content}")
        auth_instance = get_logged_in_user_email()
        tab = {}
        for i in range(len(post.files)):
            tab[i] = i
        tab = st.tabs([f"File {i+1}" for i in range(len(post.files))])
        for i in range(len(post.files)):
            if get_file_extension(post.files[i]) == "csv":
                df = pd.read_csv(post.files[i])
            elif get_file_extension(post.files[i]) == "xlsx":
                df = pd.read_excel(post.files[i])
            elif get_file_extension(post.files[i]) == "json":
                df = pd.read_json(post.files[i])
            elif get_file_extension(post.files[i]) == "sql":
                df = pd.read_sql(post.files[i])
            else:
                df = None
            with tab[i]:
                pyg_app = StreamlitRenderer(df)
                pyg_app.explorer(default_tab="data", height=730)

        handle_comment(post, auth_instance)


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
                if title and content and files:
                    create_post(
                        title,
                        content,
                        auth_instance.LoginUser(),
                        datetime.now(),
                        files,
                        image,
                    )
                    st.rerun()
                else:
                    st.error("Title, content, and files are required.")

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
