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
from utils import image_with_name
from st_click_detector import click_detector

im = Image.open("image/logo-1.png")


st.set_page_config(
    page_title="Kogga: Your home for Machine Learning",
    page_icon=im,
    layout="wide",
    initial_sidebar_state="expanded",
)

from modules import Visualization, Info, Statistic , Regression, Classification, Clustering, Post, Decomposition
from components import Navbar, Footer, Content

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)



@st.cache_data
def load_data(file):
    file_extension = os.path.splitext(file.name)[1].lower()
    if file_extension == '.csv':
        return pd.read_csv(file)
    elif file_extension in ['.xlsx', '.xls']:
        return pd.read_excel(file)

@st.experimental_dialog("Profile")
def show_profile():
    st.write("fuck")

def get_current_login():
    auth_instance = get_logged_in_user_email()
    col1, col2 = st.columns([1,3])
    with col1:
        content = f"""
            <a href='#' id='Image 1'><img width='70%' src='{image_with_name(auth_instance.LoginUser().name)}'></a>
        """
        clicked = click_detector(content)
        if clicked:
            show_profile()
    with col2:
        st.markdown(f"### {auth_instance.LoginUser().name}")

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
        selected = option_menu(None, ["Infomation", "Statistic", "Visualization", "Decomposition", "Regression", "Classification", "Clustering", "Datasets"],
                               icons=['clipboard-data', 'table', "bar-chart-fill","grid-1x2-fill" , 'rulers', 'diamond-half', 'bi-exclude','database'],
                               menu_icon="cast", default_index=0, styles={
                "st": {"padding": "5!important", "background-color": "#fafafa"},
                "icon": {"color": "black", "font-size": "15px"},
                "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px",
                             "--hover-color": "#eee"},
            })
        st.sidebar.markdown("---")
        st.markdown("#### Upload your data ####")
        file = st.file_uploader("", type=["csv", "xlsx", "xls"])

    with st.container():
        if file is not None:

            data = load_data(file)

            if selected == 'Infomation':
                Info.info(data)

            if selected == 'Statistic':
                Statistic.analyze_data(data)

            if selected == 'Visualization':
                Visualization.run(data)
            
            if selected == 'Decomposition':
                Decomposition.run(data)

            if selected == 'Regression':
                Regression.run(data)

            if selected == 'Classification':
                Classification.run(data)

            if selected == 'Clustering':
                Clustering.run(data)
        
        else:
            if selected == 'Datasets':
                Post.all_post()
            else:
                Content.content()
                Footer.footer()
            


if __name__ == "__main__":
    main()
