import pyrebase
import streamlit as st
from model import User

config = {
  "apiKey": "AIzaSyBEmyCm5XFRKRza0ZFquPbamOqwemao2XU",
  "authDomain": "kogga-f016c.firebaseapp.com",
  "projectId": "kogga-f016c",
  "databaseURL": "https://kogga-f016c.firebaseapp.com",
  "storageBucket": "kogga-f016c.appspot.com",
  "messagingSenderId": "501787704124",
  "appId": "1:501787704124:web:49ba6123e9e6d65a8b3c39",
  "measurementId": "G-YYJPPJ6PR4"
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
    auth.update_profile(user['idToken'], display_name=name, photo_url="https://firebasestorage.googleapis.com/v0/b/streamlit-ml.appspot.com/o/default_avatar.jpg?alt=media&token=904dcf59-b972-43dd-9976-beba0525783e")
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


