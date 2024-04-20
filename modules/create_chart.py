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

class Chart:
    def create_chart(chart_type, data):
        col1, col2 = st.columns(2)
        if chart_type == "Bar":
        
            st.header("Bar Chart")
            with col1:
                x_column = st.selectbox("Chọn trục X", data.columns)
            with col2:
                y_column = st.selectbox("Chọn trục Y", data.columns)
            fig = px.bar(data, x=x_column, y=y_column,color = x_column)
            st.plotly_chart(fig, theme=None, use_container_width=True)

        elif chart_type == "Line":
            st.header("Line Chart")
            multiple = st.checkbox("Vẽ nhiều đường", value=False)
            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("Chọn trục X", data.columns)

            if multiple:
                container = st.empty()
                number = container.number_input("Nhập số lượng đường", min_value=1, step=1, value=1)
                with col2:
                    y_columns = []
                    for i in range(number):
                        with col2:
                            y_column = st.selectbox(f"Chọn trục Y {i + 1}", data.columns)
                            y_columns.append(y_column)

                # Create line chart with multiple lines
                fig = px.line(data, x=x_column, y=y_columns, markers=True)

            else:
                with col2:
                    y_column = st.selectbox("Chọn trục Y", data.columns)

                # Create line chart with single line
                fig = px.line(data, x=x_column, y=y_column, markers=True)

            st.plotly_chart(fig, theme=None, use_container_width=True)

        elif chart_type == "Scatter":

            st.header("Scatter Chart")
            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("Chọn trục X", data.columns)
            with col2:
                y_column = st.selectbox("Chọn trục Y", data.columns)
            fig = px.scatter(data, x=x_column, y=y_column, color=x_column)
            st.plotly_chart(fig, theme=None, use_container_width=True)

        elif chart_type == "Pie":

            st.header("Biểu đồ tròn")
            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("Chọn nhãn", data.columns)
            with col2:
                y_column = st.selectbox("Chọn giá trị", data.columns)
            donut = st.checkbox('Sử dụng donut', value=True)
            if donut:
                # compute and show the sample statistics
                hole = 0.4

            else:
                # compute and show the population statistics
                hole = 0
            fig = px.pie(data, names=x_column, values=y_column, hole=hole)
            st.plotly_chart(fig, theme=None, use_container_width=True)

        elif chart_type == "Boxplot":
            st.header("Biểu đồ Hộp")
            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("Chọn trục X", data.columns)
            with col2:
                y_column = st.selectbox("Chọn trục Y", data.columns)

            fig = px.box(data, x=x_column, y=y_column, )
            st.plotly_chart(fig, theme=None, use_container_width=True)

        elif chart_type == "Heatmap":
            st.header("Biểu đồ nhiệt")
            choose_columns = st.multiselect("Chọn cột", data.columns)
            if not choose_columns:
                st.warning("Vui lòng chọn ít nhất 1 cột")
            else:
                all_columns = data.columns.tolist()
                columns_to_drop = [col for col in all_columns if col not in choose_columns]
                data_cp = data.drop(columns=columns_to_drop)
                data_cp = pd.get_dummies(data_cp)
                fig = px.imshow(data_cp.corr(), text_auto=True)
                fig.update_layout(autosize=False, width=600, height=600)
                st.plotly_chart(fig, theme=None, use_container_width=True)

