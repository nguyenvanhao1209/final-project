import firebase_admin
from firebase_admin import credentials, firestore
from dataclasses import asdict
from model import Comment
import streamlit as st

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate('streamlit-ml-f44ba64799b5.json')
    firebase_admin.initialize_app(cred)

db = firestore.client()

def create_comment(content, user, datetime, post):
    comment = Comment(content=content, user=user, datetime=datetime, post=post)
    comment_dict = asdict(comment)  # Convert the Comment object to a dictionary
    comment_dict['user'] = asdict(comment.user)  # Convert the User object to a dictionary
    comment_dict['post'] = asdict(comment.post)  # Convert the Post object to a dictionary

    db.collection('comments').add(comment_dict)  # Add the comment to the 'comments' collection in Firestore

def list_comment(post):
    comments = db.collection('comments').where('post', '==', post).stream()
    return [comment.to_dict() for comment in comments]
