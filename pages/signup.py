import streamlit as st
from authentication import Auth


st.set_page_config(
    page_title="Streamlit App",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="expanded",
)

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

def register():
    col1, col2, col3 = st.columns([1,5,1])
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