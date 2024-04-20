import streamlit as st
import firebase_admin
from firebase_admin import auth, exceptions, credentials, initialize_app
import asyncio
from httpx_oauth.clients.google import GoogleOAuth2
from model import User
from utils import get_name_email
from authentication import Auth

cred = credentials.Certificate("streamlit-ml-f44ba64799b5.json")

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
    st.markdown(f'<a href="{authorization_url}" target="_self">Login</a>', unsafe_allow_html=True)