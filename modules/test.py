import streamlit as st

class Test:
    def run(data):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.dataframe(data)
        with col2:
            st.dataframe(data)