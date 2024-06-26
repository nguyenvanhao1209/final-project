import streamlit as st
import base64


logo_path = r'image/logo-no-background.png'
with open(logo_path, "rb") as image_file:
    encoded_logo = base64.b64encode(image_file.read()).decode('utf-8')

class Footer:
    def footer():
        with open("style.css") as css:
            st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)
        
        footer_html = f"""
        <div class="footer">
            <div class="footer-container">
                <div class="footer-column">
                    <div class="footer-logo">
                        <img src="data:image/png;base64,{encoded_logo}" alt="logo" />
                    </div>
                    <div class="footer-social">
                        <a href="https://twitter.com" target="_blank"><i class="fa-brands fa-github"></i></a>
                        <a href="https://linkedin.com" target="_blank"><i class="fab fa-linkedin-in"></i></a>
                        <a href="https://facebook.com" target="_blank"><i class="fab fa-facebook-f"></i></a>
                    </div>
                </div>
                <div class="footer-column">
                    <h4>Product</h4>
                    <a href="#">Statistic</a>
                    <a href="#">Visualization</a>
                    <a href="#">Regression</a>
                    <a href="#">Classification</a>
                    <a href="#">Clustering</a>
                </div>
                <div class="footer-column">
                    <h4>Documentation</h4>
                    <a href="https://docs.streamlit.io/">Streamlit</a>
                    <a href="https://pandas.pydata.org/docs/">Pandas</a>
                    <a href="https://scikit-learn.org/stable/api/index.html">Scikit-learn</a>
                    <a href="https://dash.plotly.com/">Plotly</a>
                    <a href="https://docs.scipy.org/doc/scipy/">SciPy</a>
                </div>
                <div class="footer-column">
                    <h4>Company</h4>
                    <a href="#">Our Team</a>
                    <a href="#">Contact Us</a>
                    <a href="#">Host a Competition</a>
                    <a href="#">Terms · Privacy Policy</a>
                </div>
            </div>
            
        </div>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        """
        st.markdown(footer_html, unsafe_allow_html=True)