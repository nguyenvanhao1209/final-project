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
import pandas as pd

st.set_page_config(
    page_title="Streamlit App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from modules import Chart, Info, Regression, Classification, Clustering, Post
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
        st.markdown("#### Select options ####")
        selected = option_menu(None, ["Infomation", "Statistic", "Visualization", "Regression", "Classification", "Clustering", "Datasets"],
                               icons=['clipboard-data', 'table', "bar-chart-fill", 'rulers', 'diamond-half', 'bi-exclude','database'],
                               menu_icon="cast", default_index=0, styles={
                "st": {"padding": "5!important", "background-color": "#fafafa"},
                "icon": {"color": "black", "font-size": "15px"},
                "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px",
                             "--hover-color": "#eee"},
            })

    with st.container():
        
        with st.sidebar:
            st.sidebar.markdown("---")
            st.markdown("#### Upload your data ####")
            file = st.file_uploader("", type=["csv", "xlsx", "xls"])

        if file is not None:

            data = load_data(file)

            if selected == 'Infomation':
                Info.info(data)

            if selected == 'Statistic':
                analyze_data(data)

            if selected == 'Visualization':
                st.write(" # Trực quan hóa dữ liệu # ")
                st.write("#### Dữ liệu ####")
                st.write("Data")
                with st.expander("See data", expanded=True):
                    edit_data = st.data_editor(data, use_container_width=True, num_rows="dynamic")
                create_chart(edit_data)
                st.markdown("---")

            if selected == 'Regression':
                Regression.run(data)

            if selected == 'Classification':
                Classification.run(data)

            if selected == 'Clustering':
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
                Post.all_post()
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
