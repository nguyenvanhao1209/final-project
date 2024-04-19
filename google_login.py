import streamlit as st
import firebase_admin
from firebase_admin import auth, exceptions, credentials, initialize_app
import asyncio
from httpx_oauth.clients.google import GoogleOAuth2
from model import User
from utils import get_name_email
from authentication import Auth
import streamlit.components.v1 as components

cred = credentials.Certificate("streamlit-fami-eb9b5cf03485.json")

try:
    firebase_admin.get_app()
except ValueError as e:
    initialize_app(cred)
    
# Initialize Google OAuth2 client
client_id = st.secrets["client_id"]
client_secret = st.secrets["client_secret"]
redirect_url = "http://localhost:8501/home/"  # Your redirect URL

client = GoogleOAuth2(client_id=client_id, client_secret=client_secret)

async def get_access_token(client: GoogleOAuth2, redirect_url: str, code: str):
    return await client.get_access_token(code, redirect_url)

@st.cache_data
def get_logged_in_user_email():
    auth_instance = Auth()
    try:
        code = st.query_params.get('code')
        if code:
            token = asyncio.run(get_access_token(client, redirect_url, code))
            if token:
                name, user_email = get_name_email(token['id_token'])
                if user_email:
                    try:
                        user = auth.get_user_by_email(user_email)
                    except exceptions.FirebaseError:
                        user = auth.create_user(email=user_email, email_verified=True, display_name=name)
                    auth_instance.setLoginUser(User(user.uid, name, user.email))
                    return auth_instance
        return auth_instance
    except:
        pass

def google_login():
    authorization_url = asyncio.run(client.get_authorization_url(
        redirect_url,
        scope=["email", "profile"],
        extras_params={"access_type": "offline"},
    ))
    
    
    custom_button = f"""
    <link
      href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css"
      rel="stylesheet"
      integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC"
      crossorigin="anonymous"
    />
    <link
      rel="stylesheet"
      type="text/css"
      href="//fonts.googleapis.com/css?family=Open+Sans"
    />
    <script
      src="https://code.jquery.com/jquery-3.2.1.slim.min.js"
      integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN"
      crossorigin="anonymous"
    ></script>
    <script
      src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js"
      integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl"
      crossorigin="anonymous"
    ></script>
    <div style="background: transparent;">
     <a href="{authorization_url}" target="_self">
        <div style="width:500px; height: 57.5px; background-color: #4285f4; border-radius: 2px; box-shadow: 0 3px 4px 0 rgba(0, 0, 0, 0.25); position: relative; cursor: pointer;">
            <div style="position: absolute; margin-top: 1px; margin-left: 1px; width: 40px; height: 40px; border-radius: 2px; background-color: #fff;">
                <img style="position: absolute; margin-top: 11px; margin-left: 11px; width: 18px; height: 18px;" src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Google_%22G%22_logo.svg/768px-Google_%22G%22_logo.svg.png" />
            </div>
            <p style="position: absolute; margin: 11px 11px 0 110px; color: #fff; font-size: 14px; letter-spacing: 0.2px; font-family: 'Roboto';">
            <b> Sign in with Google</b>
            </p>
        </div>
        </a>
    </div>
    
    """
    st.markdown(custom_button, unsafe_allow_html=True)