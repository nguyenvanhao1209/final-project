import streamlit as st
from authentication import Auth
from PIL import Image


im = Image.open("image/logo-1.png")

st.set_page_config(
    page_title="Register | Kogga",
    page_icon=im,
    layout="wide",
    initial_sidebar_state="expanded",
)

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

def register():
    image = Image.open("image/logo-no-background.png")
    col1_logo, col2_logo, col3_logo = st.columns(3)
    with col2_logo:
        col1_logo_in, col2_logo_in, col3_logo_in = st.columns(3)
        with col2_logo_in:
            new_image = image.resize((150, 75))
            st.image(new_image)
            
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        
        with st.form("register-form"):
            st.markdown("### Create an account")

            email = st.text_input("Email")
            name = st.text_input("Name")
            password = st.text_input("Password", type='password')
            repassword = st.text_input("Re-enter Password", type='password')
            
            if st.form_submit_button("Sign Up", type='primary', use_container_width=True):
                if password != repassword:
                    st.error("Passwords do not match")
                else:
                    try:
                        user = Auth.register(email, password, name)
                        st.switch_page('pages/login.py')
                    except Exception as e:
                        st.toast("Someting went wrong", icon='🚨')
                        
            st.markdown("""
                    <p>
                    Already have an account? 
                    <a href="/login" target="_self" style="cursor:pointer"><b>Login here</b></a>
                    </p>
                    """, unsafe_allow_html=True)

def main():
    register()

if __name__ == '__main__':
    main()