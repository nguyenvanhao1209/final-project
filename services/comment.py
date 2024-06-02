import firebase_admin
from firebase_admin import credentials, firestore
from dataclasses import asdict
from model import Comment
import streamlit as st
from model import User, Post
import uuid

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate('streamlit-ml-f44ba64799b5.json')
    firebase_admin.initialize_app(cred)

db = firestore.client()

def create_comment(content, user, datetime, post):
    id = str(uuid.uuid4())
    comment = Comment(id=id,content=content, user=user, datetime=datetime, post=post)
    comment_dict = asdict(comment)  # Convert the Comment object to a dictionary
    comment_dict['user'] = asdict(comment.user)  # Convert the User object to a dictionary
    comment_dict['post'] = asdict(comment.post)  # Convert the Post object to a dictionary

    db.collection('comments').add(comment_dict)  # Add the comment to the 'comments' collection in Firestore

def list_comment(post_id):
    comments = db.collection('comments').where(field_path='post.id', op_string='==', value=post_id).stream()
    comment_list = []
    for comment in comments:
        comment_dict = comment.to_dict()
        comment_dict['user'] = User(**comment_dict['user'])
        comment_dict['post'] = Post(**comment_dict['post'])
        comment_list.append(Comment(**comment_dict))

    comment_list = sorted(comment_list, key=lambda comment: comment.datetime, reverse=True)
    return comment_list

