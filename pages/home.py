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
import streamlit_shadcn_ui as ui

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
def show_profile(user):
    st.markdown(f"""
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        """, unsafe_allow_html=True)
    st.markdown(f"""
        <div
            class="max-w-2xl mx-4 sm:max-w-sm md:max-w-sm lg:max-w-sm xl:max-w-sm sm:mx-auto md:mx-auto lg:mx-auto xl:mx-auto bg-white rounded-lg text-gray-900 mb-10">
            <div class="rounded-t-lg h-32 overflow-hidden">
                <img class="object-cover object-top w-full" src='https://images.unsplash.com/photo-1549880338-65ddcdfd017b?ixlib=rb-1.2.1&q=80&fm=jpg&crop=entropy&cs=tinysrgb&w=400&fit=max&ixid=eyJhcHBfaWQiOjE0NTg5fQ' alt='Mountain'>
            </div>
            <div class="mx-auto w-32 h-32 relative -mt-16 border-4 border-white rounded-full overflow-hidden">
                <img class="object-cover object-center h-32" src='{image_with_name(user.name, 100)}' alt='Woman looking front'>
            </div>
            <div class="text-center mt-2">
                <h2 class="font-semibold">{user.name}</h2>
                <p class="text-gray-500">{user.email}</p>
            </div>
            <ul class="py-4 mt-2 text-gray-700 flex items-center justify-around">
                <li class="flex flex-col items-center justify-around">
                    <svg class="w-4 fill-current text-blue-900" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                        <path
                            d="M10 15l-5.878 3.09 1.123-6.545L.489 6.91l6.572-.955L10 0l2.939 5.955 6.572.955-4.756 4.635 1.123 6.545z" />
                    </svg>
                    <div>2k</div>
                </li>
                <li class="flex flex-col items-center justify-between">
                    <svg class="w-4 fill-current text-blue-900" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                        <path
                            d="M7 8a4 4 0 1 1 0-8 4 4 0 0 1 0 8zm0 1c2.15 0 4.2.4 6.1 1.09L12 16h-1.25L10 20H4l-.75-4H2L.9 10.09A17.93 17.93 0 0 1 7 9zm8.31.17c1.32.18 2.59.48 3.8.92L18 16h-1.25L16 20h-3.96l.37-2h1.25l1.65-8.83zM13 0a4 4 0 1 1-1.33 7.76 5.96 5.96 0 0 0 0-7.52C12.1.1 12.53 0 13 0z" />
                    </svg>
                    <div>10k</div>
                </li>
                <li class="flex flex-col items-center justify-around">
                    <svg class="w-4 fill-current text-blue-900" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                        <path
                            d="M9 12H1v6a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-6h-8v2H9v-2zm0-1H0V5c0-1.1.9-2 2-2h4V2a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v1h4a2 2 0 0 1 2 2v6h-9V9H9v2zm3-8V2H8v1h4z" />
                    </svg>
                    <div>15</div>
                </li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    col1, col2,  = st.columns([1,2])
    with col2:
        logouted = ui.button("Logout", key="logout_btn", class_name="font-bold py-2 px-4 w-40 h-12 rounded-full")
        if logouted:
            logout()

def get_current_login():
    auth_instance = get_logged_in_user_email()
    col1, col2 = st.columns([1,3])
    with col1:
        content = f"""
            <a href='#' id='Image 1'><img width='70%' src='{image_with_name(auth_instance.LoginUser().name, 45)}'></a>
        """
        clicked = click_detector(content)
        if clicked:
            show_profile(auth_instance.LoginUser())
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
        selected = option_menu(None, ["Infomation", "Statistic", "Visualization", "Regression", "Classification", "Clustering", "Datasets"],
                               icons=['clipboard-data', 'table', "bar-chart-fill", 'rulers', 'diamond-half', 'bi-exclude','database'],
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
