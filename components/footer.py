import streamlit as st

class Footer:
    def footer():
        st.markdown(
        """
        <head>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        </head>
        <style>
            footer
            {
            visibility:hidden;
            }
            .a {
                
                background-color: #f0f2f6;
                padding: 20px;
                text-align: center;
            }
            
            .icon-list {
                display: flex;
                justify-content: center;
                align-items: center;
            }

            .icon-list-item {
                margin: 10px;
                text-align: center;
                cursor: pointer;
            }

            .icon-list-item i {
                display: block;
                font-size: 20px;
                color: black;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
        st.markdown(
            """
            <div class="a">
                <h6>Liên hệ với tôi</h6>
                <div class="icon-list">
                    <div class="icon-list-item">
                        <a href="https://github.com" target="_blank">
                            <i class="fab fa-github"></i>
                        </a>
                    </div>
                    <div class="icon-list-item">
                        <a href="https://twitter.com" target="_blank">
                            <i class="fab fa-twitter"></i>
                        </a>
                    </div>
                    <div class="icon-list-item">
                        <a href="https://youtube.com" target="_blank">
                            <i class="fab fa-youtube"></i>
                        </a>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )