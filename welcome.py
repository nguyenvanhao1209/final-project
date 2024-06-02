import streamlit as st
import streamlit.components.v1 as components
import base64
from PIL import Image
from local_components import card_container

logo_path = r'logo1.png'
with open(logo_path, "rb") as image_file:
    encoded_logo = base64.b64encode(image_file.read()).decode('utf-8')

st.set_page_config(
    page_title="Streamlit App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

st.markdown(f"""
    <header class="navbar st-emotion-cache-12fmjuu ezrtsby2">
        <a class="navbar-brand" href="#" id="nav-a">
            <img src="data:image/png;base64,{encoded_logo}" alt="logo" />
            <span>YOUR SAAS</span>
        </a>
        <a href="/login" target="_self" class="login-button">Login</a>
        <a href="/signup" target="_self" class="signup-button">Sign Up</a>
    </header>
    """, unsafe_allow_html=True)

welcome_image = Image.open("image/wimage.png")
learner = Image.open("image/learners.png")
developer = Image.open("image/developers.png")
researcher = Image.open("image/researchers.png")

col1, col2 = st.columns([3,2])

with col1:
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
    st.title("Level up with our AI & ML community")
    st.markdown(
                        """
                        <style>
                        .b {
                            margin-top: 20px ;
                            }
                        </style>

                        <div class="b"></div>
                        """,
                        unsafe_allow_html=True
                    )
    st.markdown("""<p>Join over 18M+ machine learners to share, stress test,
             and stay <br> up-to-date on all the latest ML techniques and technologies.<br> 
             Discover a huge repository of community-published models, data & code for your next project.</p>""", unsafe_allow_html=True)
    
    col1b, col2b, col3b = st.columns([1,1,4])
    with col1b:
        st.button("Get started", type='primary' ,use_container_width=True)
    with col2b:
        st.button("Learn more", type='secondary', use_container_width=True)
    st.markdown(
                        """
                        <style>
                        .b {
                            margin-top: 70px ;
                            }
                        </style>

                        <div class="b"></div>
                        """,
                        unsafe_allow_html=True
                    )    
    st.markdown("#### Who's on my app")
with col2:
    st.image(welcome_image)
    

col1n, col2n, col3n = st.columns(3)
with col1n:
    with card_container(key='learners'):
        col1ns, col2ns = st.columns([3,2])
        with col1ns:
            st.markdown("##### Learner")
            st.markdown("""<p>Dive into our courses, competitions & forums.</p>""", unsafe_allow_html=True)
        with col2ns:
            st.image(learner)
with col2n:
    with card_container(key='developers'):
        col1ns, col2ns = st.columns([3,2])
        with col1ns:
            st.markdown("##### Developer")
            st.markdown("""<p>Leverage our models, notebooks & datasets.</p>""", unsafe_allow_html=True)
        with col2ns:
            st.image(developer)
with col3n:
    with card_container(key='researcher'):
        col1ns, col2ns = st.columns([3,2])
        with col1ns:
            st.markdown("##### Researcher")
            st.markdown("""<p>Advance ML with our pre-trained model hub.</p>""", unsafe_allow_html=True)
        with col2ns:
            st.image(researcher)

