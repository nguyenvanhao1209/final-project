import streamlit as st
import pandas as pd
from PIL import Image
from streamlit_option_menu import option_menu
import os
from authentication import Auth
from services.google_login import get_logged_in_user_email
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

@st.experimental_dialog("Profile 🏴‍☠️")
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
                <li class="flex flex-col items-center justify-around">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" class="w-4 fill-current text-blue-900">
                        <path d="M2 10c0-3.967 3.69-7 8-7 4.31 0 8 3.033 8 7s-3.69 7-8 7a9.165 9.165 0 0 1-1.504-.123 5.976 5.976 0 0 1-3.935 1.107.75.75 0 0 1-.584-1.143 3.478 3.478 0 0 0 .522-1.756C2.979 13.825 2 12.025 2 10Z" clip-rule="evenodd" />
                    </svg>
                    <div>10k</div>
                </li>
                <li class="flex flex-col items-center justify-around">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" class="w-4 fill-current text-blue-900">
                        <path d="M10 1c3.866 0 7 1.79 7 4s-3.134 4-7 4-7-1.79-7-4 3.134-4 7-4Zm5.694 8.13c.464-.264.91-.583 1.306-.952V10c0 2.21-3.134 4-7 4s-7-1.79-7-4V8.178c.396.37.842.688 1.306.953C5.838 10.006 7.854 10.5 10 10.5s4.162-.494 5.694-1.37ZM3 13.179V15c0 2.21 3.134 4 7 4s7-1.79 7-4v-1.822c-.396.37-.842.688-1.306.953-1.532.875-3.548 1.369-5.694 1.369s-4.162-.494-5.694-1.37A7.009 7.009 0 0 1 3 13.179Z" clip-rule="evenodd" />
                    </svg>
                    <div>10k</div>
                </li>
                <li class="flex flex-col items-center justify-around">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" class="w-4 fill-current text-blue-900">
                        <path d="M4.93 2.31a41.401 41.401 0 0 1 10.14 0C16.194 2.45 17 3.414 17 4.517V17.25a.75.75 0 0 1-1.075.676l-2.8-1.344-2.8 1.344a.75.75 0 0 1-.65 0l-2.8-1.344-2.8 1.344A.75.75 0 0 1 3 17.25V4.517c0-1.103.806-2.068 1.93-2.207Zm4.822 3.997a.75.75 0 1 0-1.004-1.114l-2.5 2.25a.75.75 0 0 0 0 1.114l2.5 2.25a.75.75 0 0 0 1.004-1.114L8.704 8.75h1.921a1.875 1.875 0 0 1 0 3.75.75.75 0 0 0 0 1.5 3.375 3.375 0 1 0 0-6.75h-1.92l1.047-.943Z" clip-rule="evenodd" />
                    </svg>
                    <div>10k</div>
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
