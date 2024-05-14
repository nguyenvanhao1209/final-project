import streamlit as st
import pandas as pd
from PIL import Image
from streamlit_option_menu import option_menu
from streamlit_timeline import timeline
import os
import pygwalker as pyg
from authentication import Auth
from services.google_login import get_logged_in_user_email
from pygwalker.api.streamlit import StreamlitRenderer
from services.post import list_post
from PIL import Image
import requests
from io import BytesIO
from utils import time_difference, format_file_size, get_file_extension
import zipfile
import io
from utils import get_file
import pandas as pd
from services.comment import create_comment, list_comment
from datetime import datetime
from local_components import card_container

st.set_page_config(
    page_title="Streamlit App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from modules import Chart, Info, Regression, Classification, Clustering
from components import Navbar, Footer

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)





@st.cache_data
def load_data(file):
    file_extension = os.path.splitext(file.name)[1].lower()
    if file_extension == '.csv':
        return pd.read_csv(file)
    elif file_extension in ['.xlsx', '.xls']:
        return pd.read_excel(file)


def summary(df):
    summary = df.describe()
    return summary


def summary_p(df):
    summary = df.describe()
    return summary


### analyze_data      
def analyze_data(data):
    # Perform basic data analysis
    st.write(" # Data Analysis # ")
    st.write("#### Dữ liệu ####")
    st.write("Data")
    with st.expander("See data", expanded=True):
        edited_df = st.data_editor(data,use_container_width=True,num_rows="dynamic")

    st.markdown("---")
    ######
    st.write("#### Thống kê mô tả một chiều ####")

    st.markdown("###### Bảng giá trị thống kê mô tả ######")
    use_sample_stats = st.checkbox('Hiệu chỉnh mẫu thống kê', value=True)
    if use_sample_stats:
        # compute and show the sample statistics
        st.dataframe(summary(edited_df), use_container_width=True)
        st.download_button(
            label="Download data as CSV",
            data=summary(data).to_csv(index=False),
            file_name='data_analyze.csv',
            mime='text/csv')

    else:
        # compute and show the population statistics
        st.dataframe(summary_p(edited_df), use_container_width=True)
        st.download_button(
            label="Download data as CSV",
            data=summary_p(data).to_csv(index=False),
            file_name='data_analyze.csv',
            mime='text/csv')
    


#### Data visualization
def create_chart(data):
    pyg_app = StreamlitRenderer(data)
    pyg_app.explorer()
    

def get_current_login():
    auth_instance = get_logged_in_user_email()
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown(f"### {auth_instance.LoginUser().name}")
    with col2:
        if st.button('Logout', type='primary'):
            logout()

def logout():
    Auth.logout()
    st.switch_page('pages/login.py')

# main function
def main():
    Navbar.navbar()
    with st.sidebar:
        get_current_login()
           
        st.sidebar.markdown("---")
        st.markdown("#### Chọn chức năng ####")
        selected = option_menu(None, ["Dữ liệu", "Thống kê", "Trực quan hóa", "Hồi quy", "Phân lớp", "Phân cụm", "Datasets"],
                               icons=['clipboard-data', 'table', "bar-chart-fill", 'rulers', 'diamond-half', 'bi-exclude'],
                               menu_icon="cast", default_index=0, styles={
                "st": {"padding": "5!important", "background-color": "#fafafa"},
                "icon": {"color": "black", "font-size": "15px"},
                "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px",
                             "--hover-color": "#eee"},
            })

    with st.container():
        
        with st.sidebar:
            st.sidebar.markdown("---")
            st.markdown("#### Tải lên dữ liệu ####")
            file = st.file_uploader("", type=["csv", "xlsx", "xls"])

        if file is not None:

            data = load_data(file)

            if selected == 'Dữ liệu':
                
                Info.info(data)

            if selected == 'Thống kê':
                
                analyze_data(data)

            if selected == 'Trực quan hóa':
                
                st.write(" # Trực quan hóa dữ liệu # ")
                st.write("#### Dữ liệu ####")
                st.write("Data")
                with st.expander("See data", expanded=True):
                    edit_data = st.data_editor(data, use_container_width=True, num_rows="dynamic")
                create_chart(edit_data)
                st.markdown("---")

            if selected == 'Hồi quy':
                

                st.write(" # Hồi quy tuyến tính # ")
                st.write("#### Dữ liệu ####")
                st.write("Data")
                with st.expander("See data", expanded=True):
                    edit_data = st.data_editor(data, use_container_width=True, num_rows="dynamic")
                st.markdown("---")
                regression_type = st.selectbox("", ["OLS Linear Regression", 'Ridge', 'Lasso'])
                if regression_type == "OLS Linear Regression":
                    Regression.simple_linear_regresstion(data)
                if regression_type == "Ridge":
                    Regression.ridge_regression(data)
                if regression_type == "Lasso":
                    Regression.lasso_regression(data)

            if selected == 'Phân lớp':
                
                st.write(" # Phân lớp # ")
                st.write("#### Dữ liệu ####")
                st.write("Data")
                with st.expander("See data", expanded=True):
                    edit_data = st.data_editor(data, use_container_width=True, num_rows="dynamic")
                st.markdown("---")
                class_type = st.selectbox("", ["KNN", 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'SVM'])
                if class_type == 'KNN':
                    Classification.knn_classification(edit_data)
                if class_type == 'Logistic Regression':
                    Classification.lgreg_classification(edit_data)
                if class_type == 'Random Forest':
                    Classification.randomfor_classification(edit_data)
                if class_type == 'Naive Bayes':
                    Classification.naivebayes_classification(edit_data)
                if class_type == 'SVM':
                    Classification.svm_classification(edit_data)

            if selected == 'Phân cụm':
                
                st.write(" # Phân cụm # ")
                st.write("#### Dữ liệu ####")
                st.write("Data")
                with st.expander("See data", expanded=True):
                    edit_data = st.data_editor(data, use_container_width=True, num_rows="dynamic")
                st.markdown("---")
                class_type = st.selectbox("", ["K Means", 'DBSCAN', 'OPTICS'])
                if class_type == 'K Means':
                    Clustering.kmeans_clustering(edit_data)
                if class_type == 'DBSCAN':
                    Clustering.dbscan_clustering(edit_data)
                if class_type == 'OPTICS':
                    Clustering.optics_clustering(edit_data)
        
        else:
            if selected == 'Datasets':
                st.write(" # Tập dữ liệu # ")
                col1, col2 = st.columns([5,1])
                with col1:
                    st.write("Khám phá, phân tích và chia sẻ dữ liệu chất lượng")
                with col2:
                    if st.button('Upload your data', type='primary'):
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
                                st.session_state.current_post = post
                                st.switch_page("pages/detail_dataset.py")
            else:    
                st.balloons()
                container = st.container()
                with container:
                    with st.spinner(text="Building line"):
                        with open('timeline.json', "r", encoding="utf-8") as f:
                            data = f.read()
                            timeline(data, height=450, )
                st.markdown(
                    """
                    <style>
                    .b {
                        margin-top: 50px ;
                        }
                    </style>

                    <div class="b"></div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(" ### Làm sao để sử dụng ?")
                st.markdown(
                    """
                    <style>
                    .b {
                        margin-top: 50px ;
                        }
                    </style>

                    <div class="b"></div>
                    """,
                    unsafe_allow_html=True
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                                        <head>
                                        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
                                        </head>
                                        <body>

                                        <i class="fa-solid fa-1 fa-beat" style="font-size:70px;color: #ff4b4b;"></i>
                                        <h5>Tải lên dữ liệu của bạn</h5>
                                        </body>


                                        """, unsafe_allow_html=True)
                    image1 = Image.open("image/im1.png")
                    st.image(image1)

                    st.markdown(
                        """
                        <style>
                        .b {
                            margin-top: 50px ;
                            }
                        </style>

                        <div class="b"></div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown("""
                                        <head>
                                        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
                                        </head>
                                        <body>

                                        <i class="fa-solid fa-3 fa-beat" style="font-size:70px;color: #ff4b4b;"></i>
                                        <h5>Bắt đầu tính toán </h5>
                                        </body>


                                        """, unsafe_allow_html=True)
                    image3 = Image.open("image/im3.png")
                    st.image(image3)

                with col2:
                    st.markdown("""
                                        <head>
                                        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
                                        </head>
                                        <body>

                                        <i class="fa-solid fa-2 fa-beat" style="font-size:70px;color: #ff4b4b;"></i>
                                        <h5>Chọn chức năng mong muốn</h5>
                                        </body>


                                        """, unsafe_allow_html=True)
                    image2 = Image.open("image/im2.png")
                    st.image(image2)
                    st.markdown(
                        """
                        <style>
                        .b {
                            margin-top: 50px ;
                            }
                        </style>

                        <div class="b"></div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown("""
                                        <head>
                                        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
                                        </head>
                                        <body>

                                        <i class="fa-solid fa-4 fa-beat" style="font-size:70px;color: #ff4b4b;"></i>
                                        <h5>Tải xuống và tiếp tục công việc</h5>
                                        </body>


                                        """, unsafe_allow_html=True)
                    image4 = Image.open("image/im4.png")
                    st.image(image4)
                container.markdown("---")
                Footer.footer()
            


if __name__ == "__main__":
    main()
