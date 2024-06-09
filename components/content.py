import streamlit as st
from streamlit_timeline import timeline
from PIL import Image
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from services.google_login import get_logged_in_user_email
import base64




def card_metric(wch_colour_box, wch_colour_font, iconname, sline, content, i):
    lnk = '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">'

    htmlstr = f"""
    <style>
    .card-{i} {{
        background-color: rgb({wch_colour_box[0]}, {wch_colour_box[1]}, {wch_colour_box[2]});
        color: rgb({wch_colour_font[0]}, {wch_colour_font[1]}, {wch_colour_font[2]});
        border-radius: 7px;
        padding: 10px;
        transition: transform 0.2s;
    }}
    .card-{i}:hover {{
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }}
    </style>
    <div class="card-{i}" style="display: inline-block;">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <i class="{iconname}" style="font-size: 35px;"></i>
        </div>
        <div style="font-size: 20px; font-weight: bold;">{sline}</div>
        <div style="font-size: 15px;">{content}</div>
    </div>
    """
    st.markdown(lnk + htmlstr, unsafe_allow_html=True)
    



def get_step_image(icon, content , image, height, image_border=None, hover_effect=None):
    logo_path = f'image/{image}'
    with open(logo_path, "rb") as image_file:
        encoded_logo = base64.b64encode(image_file.read()).decode('utf-8')

    st.markdown(
        f"""
        <head>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
            <style>
                .step-image-container {{
                    text-align: center;
                }}
                .step-image {{
                    max-width: 100%;
                    max-height: 100%;
                    {f"width: 100%;"}
                    {f"height: {height}px;"}
                    {f"border: {image_border}px solid #ff4b4b;"}
                    {'transition: transform 0.2s;' if hover_effect else ""}
                }}
                .step-image:hover {{
                    {f"transform: scale({hover_effect});" if hover_effect else ""}
                }}
            </style>
        </head>
        <body>
            <div class="step-image-container">
                <i class="fa-solid {icon}" style="font-size:50px;color: #ff4b4b;"></i>
                <h6>{content}</h6>
                <img class="step-image" src="data:image/png;base64,{encoded_logo}" alt="logo" />
            </div>
        </body>
        """,
        unsafe_allow_html=True,
    )


def margin(pixel):
    st.markdown(
        f"""
            <div style="margin-top: {pixel}px"></div>
            """,
        unsafe_allow_html=True,
    )


class Content:
    def content():
        st.balloons()
        user = get_logged_in_user_email()
        st.markdown(f" ##### Welcome to kogga, {user.LoginUser().name}!")
        st.markdown(
        """
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
        <style>
        .dataset-card {
            display: flex;
            align-items: center;
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            background-color: #fff;
            width: 240px;
            margin: 20px;
        }
        .dataset-icon {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 50px;
            width: 50px;
            border-radius: 50%;
            background-color: #FFD6D6;
            margin-right: 10px;
        }
        .dataset-icon i {
            font-size: 24px;
            color: #ff4b4b;
        }
        .dataset-info {
            font-family: 'Segoe UI', sans-serif;
        }
        .dataset-info h3 {
            margin: 0;
            font-size: 18px;
            font-weight: bold;
        }
        .dataset-info p {
            margin: 0;
            font-size: 12px;
            color: #555;
        }
        </style>
        """, unsafe_allow_html=True
    )

        col1_in, col2_in, col3_in = st.columns([1,1,2])
        with col1_in:
            st.markdown(
                """
                <div class="dataset-card">
                    <div class="dataset-icon">
                        <i class="fas fa-database"></i>
                    </div>
                    <div class="dataset-info">
                        <h3>Datasets</h3>
                        <p>1 total created</p>
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
            
            st.markdown(
                """
                <div class="dataset-card">
                    <div class="dataset-icon">
                        <i class="fas fa-database"></i>
                    </div>
                    <div class="dataset-info">
                        <h3>Datasets</h3>
                        <p>1 total created</p>
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
        with col2_in:
            st.markdown(
                """
                <div class="dataset-card">
                    <div class="dataset-icon">
                        <i class="fas fa-database"></i>
                    </div>
                    <div class="dataset-info">
                        <h3>Datasets</h3>
                        <p>1 total created</p>
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
            
            st.markdown(
                """
                <div class="dataset-card">
                    <div class="dataset-icon">
                        <i class="fas fa-database"></i>
                    </div>
                    <div class="dataset-info">
                        <h3>Datasets</h3>
                        <p>1 total created</p>
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
        with col3_in:
            st.markdown(
            """
            <style>
            .stat-container {
                display: flex;
                justify-content: space-between;
                width: 100%;
                height: 220px;
            }
            .stat-card {
                display: flex;
                flex-direction: column;
                align-items: center;
                width: 45%;
                padding: 20px;
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                background-color: #fff;
                margin-left: 10px;
            }
            .stat-card h3 {
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
                font-weight: bold;
                margin-bottom: 10px;
            }
            .stat-card p {
                font-family: 'Segoe UI', sans-serif;
                font-size: 24px;
                margin: 0;
            }
            .stat-card .subtext {
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
                color: #555;
                margin-top: 5px;
            }
            .circle-progress {
                position: relative;
                display: inline-block;
            }
            .circle-progress svg {
                transform: rotate(-90deg);
            }
            .circle-progress .progress-value {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                font-family: 'Segoe UI', sans-serif;
                font-size: 24px;
            }
            .circle-progress .progress-text {
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
                color: #555;
                text-align: center;
                margin-top: 5px;
            }
            </style>
            """, unsafe_allow_html=True
        )

        # Helper function to create the circular progress bar
            def circular_progress(percent):
                return f"""
                <div class="circle-progress">
                    <svg width="100" height="100">
                        <circle cx="50" cy="50" r="45" stroke="#e0e0e0" stroke-width="10" fill="none"/>
                        <circle cx="50" cy="50" r="45" stroke="#28a745" stroke-width="10" fill="none" stroke-dasharray="{percent * 2.83} 283" />
                    </svg>
                    <div class="progress-value">{percent}%</div>
                </div>
                """

            # Main layout
            st.markdown(
                """
                <div class="stat-container">
                    <div class="stat-card">
                        <h3>LOGIN STREAK</h3>
                        <p>1</p>
                        <div class="subtext">Your longest is 5 days,</div>
                    </div>
                    <div class="stat-card">
                        <h3>TIER PROGRESS</h3>
                        <div class="circle-progress">
                            <svg width="100" height="100">
                                <circle cx="50" cy="50" r="45" stroke="#ff4b4b" stroke-width="10" fill="none"/>
                                <circle cx="50" cy="50" r="45" stroke="#ff4b4b" stroke-width="10" fill="none" stroke-dasharray="{percent * 2.83} 283" />
                            </svg>
                            <div class="progress-value"> 50% </div>
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
        st.markdown("---")            
        st.markdown("""<h5> What do we have ?</h5>""", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            card_metric([255,164,33], [256,256,256], "fa-solid fa-laptop-code","50+","We provide state-of-the-art machine learning algorithms",1)
        with col2:
            card_metric([0,192,242], [256,256,256], "fa-solid fa-users","10+","We have strong communities with interesting members" ,2)
        with col3:
            card_metric([128,61,245], [256,256,256], "fa-solid fa-database","10+", "We have lots of great and useful datasets that you can use for free",124)
        st.markdown("---")
        st.markdown(" ##### How to use my app")
        col_step = st.columns(4)
        margin(70)
        with col_step[0]:
            get_step_image("fa-file-upload","Upload your data", "step1.png",200, image_border=0.1, hover_effect=1.1)
        with col_step[1]:
            get_step_image("fa-desktop","Choose your ml method", "step2.png",200, image_border=0.1, hover_effect=1.1)
        with col_step[2]:
            get_step_image("fa-fill-drip","Fill your paramater", "step3.png",200, image_border=0.1, hover_effect=1.1)
        with col_step[3]:
            get_step_image("fa-square-poll-vertical","Get the result", "step4.png",200, image_border=0.1, hover_effect=1.1)
            
        
        
        
        
        
        
            
            
            
            
        
