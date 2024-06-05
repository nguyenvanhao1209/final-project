from typing import Container
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
import streamlit_shadcn_ui as ui
from local_components import card_container
from streamlit_shadcn_ui import slider, input, textarea, radio_group, switch


class Info:

    def info(data):
        content_image = Image.open("image/content1.png")
        col1, col2 = st.columns([5,1])
        with col1:
            st.markdown(" # Data infomation # ")
            st.markdown("""<p>Detail infomation of dataset </p>""", unsafe_allow_html=True)
        with col2:
            st.image(content_image)
        st.markdown("---")
        st.write("#### Your data ####")
        filtered_df = dataframe_explorer(data, case=False)
        st.dataframe(filtered_df, use_container_width=True)
        st.download_button(
            label="Download filter data",
            data=filtered_df.to_csv(index=False),
            file_name='data_filter.csv',
            mime='text/csv',
        )
        st.markdown("---")

        st.write("#### Detail information ####")
        r = data.shape[0]
        c = data.shape[1]
        col1f, col2f = st.columns([4,1])
        with col1f:
            st.markdown(f"Data size: :red[{r}] x :red[{c}]")
        with col2f:
            data.dropna(inplace=True)
            data.reset_index(drop=True, inplace=True)

            st.download_button(
                label="Download clean data",
                data=data.to_csv(index=False),
                file_name='data_clean.csv',
                mime='text/csv',
            )
        

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.write("Columns: ")
            st.dataframe(data.columns, use_container_width=True)

        with col2:
            st.write("Data types: ")
            st.dataframe(data.dtypes, use_container_width=True)

        with col3:
            st.write("Unique Values: ")
            st.dataframe(data.nunique(), use_container_width=True)
        with col4:
            st.markdown("Missing Values: ")
            st.dataframe(data.isnull().sum(), use_container_width=True)

        
        

