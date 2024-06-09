import streamlit as st
import firebase_admin
from firebase_admin import auth, exceptions, credentials, initialize_app, firestore
from datetime import datetime

cred = credentials.Certificate("streamlit-ml-f44ba64799b5.json")

try:
    firebase_admin.get_app()
except ValueError as e:
    initialize_app(cred)

def count_users():
    users = auth.list_users()
    user_count = 0
    for user in users.iterate_all():
        user_count += 1
    return user_count

def count_posts():
    db = firestore.client()

    posts_ref = db.collection('posts')
    posts = posts_ref.get()

    return len(posts)

def count_user_posts(user_id):
    db = firestore.client()

    user_posts_ref = db.collection('posts').where('user_id', '==', user_id)
    user_posts = user_posts_ref.get()

    return len(user_posts)

def count_user_comments(user_id):
    db = firestore.client()

    user_comments_ref = db.collection('comments').where('user.id', '==', user_id)
    user_comments = user_comments_ref.get()

    return len(user_comments)

def count_user_votes(user_id):
    db = firestore.client()

    user_votes_ref = db.collection('votes').where('user.id', '==', user_id)
    user_votes = user_votes_ref.get()

    return len(user_votes)

def count_comments_in_user_posts(user_id):
    db = firestore.client()

    user_posts_ref = db.collection('posts').where('user_id', '==', user_id)
    user_posts = user_posts_ref.get()

    total_comments = 0
    for post in user_posts:
        post_comments_ref = db.collection('comments').where('post.id', '==', post.id)
        post_comments = post_comments_ref.get()
        total_comments += len(post_comments)

    return total_comments

def get_last_login(user_id):
  user = auth.get_user(user_id)
  last_login = user.user_metadata.last_sign_in_timestamp

  if last_login is None:
    return 1
  else:
    # Convert the timestamp from milliseconds to seconds
    last_login_seconds = last_login / 1000

    timestamp_datetime = datetime.fromtimestamp(last_login_seconds)

    # Extract the date
    date = timestamp_datetime.date()

    # Get the present date
    present_date = datetime.now().date()

    # Calculate the number of days between the date and the present
    days = (present_date - date).days

    return days
