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

from modules import Chart, Info, Statistic , Regression, Classification, Clustering, Post
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
                Statistic.analyze_data(data)

            if selected == 'Visualization':
                content_image = Image.open("image/content2.png")
                col1, col2 = st.columns([3,1])
                with col1:
                    st.markdown(" # Visualization # ")
                    st.markdown("""<p>Easy to visualize your data with drag and drop</p>""", unsafe_allow_html=True)
                with col2:
                    st.image(content_image)
                st.markdown("---")
                st.write("#### Your data ####")
                with st.expander("See data", expanded=True):
                    edit_data = st.data_editor(data, use_container_width=True, num_rows="dynamic")
                create_chart(edit_data)
                st.markdown("---")

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
