import pyrebase
import streamlit as st
from model import User

config = {
  "apiKey": "AIzaSyA3i-QDbnc6Qvw9Q9fZlmOdLUhR9g-z7uE",
  "authDomain": "streamlit-ml.firebaseapp.com",
  "projectId": "streamlit-ml",
  "databaseURL": "https://streamlit-ml.firebaseapp.com",
  "storageBucket": "streamlit-ml.appspot.com",
  "messagingSenderId": "765035265901",
  "appId": "1:765035265901:web:7debe2112496d4a1e0a464",
  "measurementId": "G-5T0EV40LMX"
}

firebase = pyrebase.initialize_app(config)
auth = firebase.auth()

class Auth:

  def __init__(self):
    self.current_user = None

  def sign_in(email, password):
    return auth.sign_in_with_email_and_password(email, password)

  def register(email, password, name):
    user = auth.create_user_with_email_and_password(email, password)
    auth.update_profile(user['idToken'], display_name=name)
    return user
  
  def logout():
     auth.current_user = None
     return
  
  def setLoginUser(self, user: User):
     self.current_user = user

  def LoginUser(self):
    user = auth.current_user
    if user:
        return User(user['localId'], user['displayName'], user['email'])
    else:
        return self.current_user


