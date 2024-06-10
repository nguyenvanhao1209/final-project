import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from streamlit_extras import F
from streamlit_option_menu import option_menu
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_timeline import timeline
import numpy as np
import math
import os
import plotly.graph_objects as go
import scipy.stats as stats
from scipy.stats import t
from scipy.stats import chi2_contingency
import plotly.figure_factory as ff
from pygwalker.api.streamlit import StreamlitRenderer

class Visualization:
    #### Data visualization
    def create_chart(data):
        pyg_app = StreamlitRenderer(data)
        pyg_app.explorer()
        
    def run(data):
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
        Visualization.create_chart(edit_data)
        st.markdown("---")
        
        

