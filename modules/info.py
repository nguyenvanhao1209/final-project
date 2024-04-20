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
        st.write(" # Thông tin dữ liệu # ")
        st.write("#### Dữ liệu ####")
        filtered_df = dataframe_explorer(data, case=False)
        st.dataframe(filtered_df, use_container_width=True)
        st.download_button(
            label="Download filter data",
            data=filtered_df.to_csv(index=False),
            file_name='data_filter.csv',
            mime='text/csv',
        )
        st.markdown("---")

        st.write("#### Thông tin ####")
        r = data.shape[0]
        c = data.shape[1]
        st.markdown(f"Kích thước dữ liệu: :red[{r}] x :red[{c}]")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("Tên cột: ")
            st.dataframe(data.columns, use_container_width=True)

        with col2:
            st.write("Kiểu dữ liệu cột: ")
            st.dataframe(data.dtypes, use_container_width=True)

        with col3:
            st.write("Unique Values: ")
            st.dataframe(data.nunique(), use_container_width=True)

        st.markdown("---")

        st.markdown("#### Missing Values ####")
        col1n, col2n = st.columns([2, 3])
        with col1n:
            st.dataframe(data.isnull().sum(), use_container_width=True)
        with col2n:
            st.markdown(
                "   Missing value (giá trị thiếu) trong khoa học dữ liệu và phân tích dữ liệu là giá trị không tồn tại hoặc không được xác định trong tập dữ liệu.")
            st.markdown(
                "Missing value có thể xảy ra khi dữ liệu bị mất hoặc bị lỗi trong quá trình thu thập hoặc nhập liệu, hoặc có thể do một giá trị không có ý nghĩa được đại diện bởi các giá trị như NaN (Not a Number) hoặc NULL trong các ngôn ngữ lập trình.")
            data.dropna(inplace=True)
            data.reset_index(drop=True, inplace=True)

            st.download_button(
                label="Download clean data",
                data=data.to_csv(index=False),
                file_name='data_clean.csv',
                mime='text/csv',
            )

