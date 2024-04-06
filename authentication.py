import pyrebase
import streamlit as st
from model import User

config = {
  "apiKey": "AIzaSyB4o8GQ3M-n_JCCjtBv9p9WcmnH5Q0ZDyE",
  "authDomain": "streamlit-fami.firebaseapp.com",
  "projectId": "streamlit-fami",
  "databaseURL": "https://streamlit-fami.firebaseapp.com",
  "storageBucket": "streamlit-fami.appspot.com",
  "messagingSenderId": "785453725430",
  "appId": "1:785453725430:web:2875bfbc8834df51891ed7",
  "measurementId": "G-3C13TD2BEG"
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


