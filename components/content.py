import streamlit as st
from streamlit_timeline import timeline
from PIL import Image


with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)


def get_step_image(icon, image):
    st.markdown(
        f"""
                                <head>
                                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
                                </head>
                                <body>

                                <i class="fa-solid {icon} fa-beat" style="font-size:70px;color: #ff4b4b;"></i>
                                <h5>Tải lên dữ liệu của bạn</h5>
                                </body>


                                """,
        unsafe_allow_html=True,
    )
    image_render = Image.open(f"""image/{image}""")
    st.image(image_render)


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
        with st.container():
            with st.spinner(text="Building line"):
                with open("timeline.json", "r", encoding="utf-8") as f:
                    data = f.read()
                    timeline(
                        data,
                        height=450,
                    )
        margin(50)
        
        st.container().markdown("---")

        st.markdown(" ### How to use ?")
        margin(50)

        col1, col2 = st.columns(2)
        with col1:
            get_step_image("fa-1", "im1.png")
            margin(50)
            get_step_image("fa-3", "im3.png")
        with col2:
            get_step_image("fa-2", "im2.png")
            margin(50)
            get_step_image("fa-4", "im4.png")
        
        st.markdown("---")
            
        
