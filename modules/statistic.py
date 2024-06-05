import streamlit as st
import pandas as pd
from PIL import Image
import pandas as pd
import math
from local_components import card_container

def summary(df):
    summary = df.describe()
    return summary


def summary_p(df):
    n = df.shape[0]
    summary = df.describe()
    summary.loc['std'] = summary.loc['std']*math.sqrt((n-1)/n)
    return summary

class Statistic:
    def analyze_data(data):
    # Perform basic data analysis
        content_image = Image.open("image/content7.png")
        col1, col2 = st.columns([4,1])
        with col1:
            st.markdown(" # Data Analysis # ")
            st.markdown("""<p>Descirib analyze your data and online compute statistic </p>""", unsafe_allow_html=True)
        with col2:
            st.image(content_image)
        st.markdown("---")
        st.write("#### Your data ####")
        with st.expander("See data", expanded=True):
            edited_df = st.data_editor(data,use_container_width=True,num_rows="dynamic")

        st.markdown("---")
        ######
        
        st.markdown("###### Descriptive statistics table ######")
        use_sample_stats = st.checkbox('Calibrate statistical samples', value=True)
        if use_sample_stats:
            # compute and show the sample statistics
            st.dataframe(summary(edited_df), use_container_width=True)
            st.download_button(
                label="Download data as CSV",
                data=summary(data).to_csv(index=False),
                file_name='data_analyze.csv',
                mime='text/csv')

        else:
            # compute and show the population statistics
            st.dataframe(summary_p(edited_df), use_container_width=True)
            st.download_button(
                label="Download data as CSV",
                data=summary_p(data).to_csv(index=False),
                file_name='data_analyze.csv',
                mime='text/csv')
        st.markdown("---")
        st.markdown("###### Online statistic caculator ######")
        col1s, col2s = st.columns(2)
        with col1s:
            with card_container(key="mean-form"):
                text_means = """
                is a statistical concept used to measure the centrality of a data set.\n
                It is calculated by taking the sum of all the values ​​in the data set and dividing by the number of those values.
                """
                st.markdown("###### Mean ######")
                col1_inside, col2_inside = st.columns([100,1])
                with col1_inside:
                    input1 = st.text_input("Example:",placeholder="Ex: 1,2,4,2,5", help=text_means)


                values = input1.split(',')


                numeric_values = []
                for value in values:
                    value = value.strip()
                    if value:
                        numeric_values.append(float(value))


                series1 = pd.Series(numeric_values)


                series1_mean = series1.mean()

                if input1:
                    st.markdown(f"Mean: <span style='color:green;'>{series1_mean}</span>", unsafe_allow_html=True)
            
            with card_container(key="quatiles-form"):
                text_quatiles ="""
                is a descriptive statistical method for dividing a data set into four equal parts.\n
                The quartiles divide the data set into three intervals, numbered Q1 to Q3, such that the interval between Q1 and Q3 contains 50% of the data and the interval between Q2 (i.e., the median) also contains 50% data.
                - (Q1) corresponds to the 25% percentile
                - (Q2) corresponds to the 50% percentile or median value
                - (Q3) corresponds to the 75% percentile
                """
                st.markdown("###### Quartiles ######")
                col1_inside, col2_inside = st.columns([100,1])
                with col1_inside:
                    input3 = st.text_input("Example:",placeholder="Ex: 1,2,4,2,5", help=text_quatiles)
                values3 = input3.split(',')
                numeric_values3 = []
                for value3 in values3:
                    value3 = value3.strip()
                    if value3:
                        numeric_values3.append(float(value3))
                series3 = pd.Series(numeric_values3)
                q1 = series3.quantile(0.25)
                q2 = series3.quantile(0.5)  # Median
                q3 = series3.quantile(0.75)
                if input3:
                    st.markdown(f"Q1: <span style='color:green;'>{q1}</span>", unsafe_allow_html=True)
                    st.markdown(f"Q2 (Median): <span style='color:green;'>{q2}</span>", unsafe_allow_html=True)
                    st.markdown(f"Q3: <span style='color:green;'>{q3}</span>", unsafe_allow_html=True)
                        
            
            
        with col2s:
            with card_container(key="variance-form"):
                text_variance = """
                is a measure of the dispersion of values ​​in a data set.\n
                It measures the deviation of each value from the mean of that data set. Variance is calculated by taking the sum of the squares of the difference between each value and the mean, divided by the number of values ​​in the data set.
                """
                st.markdown("###### Variance ######")
                col1_inside, col2_inside = st.columns([100,1])
                with col1_inside:
                    input2 = st.text_input("Example:",placeholder="Ex: 1,2,4,2,5", help=text_variance)
                values2 = input2.split(',')
                numeric_values1 = []
                for value in values2:
                    value = value.strip()
                    if value:
                        numeric_values1.append(float(value))
                series2 = pd.Series(numeric_values1)
                series2_var = series2.var()
                if input1:
                    st.markdown(f"Variance: <span style='color:green;'>{series2_var}</span>", unsafe_allow_html=True)
                    
            with card_container(key="skewness-form"):
                text_skewness = """
                is a statistical measure used to measure the degree of asymmetry of data distribution.\n
                It measures the deviation of the data distribution from a normal or symmetric distribution
                """
                st.markdown("###### Skewness ######")
                col1_inside, col2_inside = st.columns([100,1])
                with col1_inside:
                    input4 = st.text_input("Example: ",placeholder="Ex: 1,2,4,2,5", help=text_skewness)
                values4 = input4.split(',')
                numeric_values4 = []
                for value4 in values4:
                    value4 = value4.strip()
                    if value4:
                        numeric_values4.append(float(value4))
                series4 = pd.Series(numeric_values4)
                skewness = series4.skew()
                if input4:
                    st.markdown(f"Skewness: <span style='color:green;'>{skewness}</span>", unsafe_allow_html=True)
                        
        
        