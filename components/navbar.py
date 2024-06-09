import streamlit as st
import base64


logo_path = r'image/logo-no-background.png'
with open(logo_path, "rb") as image_file:
    encoded_logo = base64.b64encode(image_file.read()).decode('utf-8')

class Navbar:
    def navbar():
        with open("style.css") as css:
            st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <head>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
            <script src="https://cdn.tailwindcss.com"></script>
            <style>
            </style>
            </head>
            <body>
            
            
            <header class="navbar-home st-emotion-cache-12fmjuu ezrtsby2">
                <a class="logo-image" href="#" id="nav-a">
                    <img src="data:image/png;base64,{encoded_logo}" alt="logo" />
                </a>         
                <form class="search-form" action="" style="margin:auto;max-width:800px">
                    <input type="text" placeholder="     Search..." name="search2" />
                    <button type="submit"><i class="fa fa-search"></i></button>
                </form>
            </header>
            </body>
            
            
            """,
            unsafe_allow_html=True,
        )
        
        
