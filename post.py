import firebase_admin
from firebase_admin import credentials, firestore, initialize_app
from model import Post, User

cred = credentials.Certificate("streamlit-ml-f44ba64799b5.json")

try:
    firebase_admin.get_app()
except ValueError as e:
    initialize_app(cred)
    
def create_post(title, content, author, date, file):
    db = firestore.client()

    # Create a new Post instance
    post = Post(title=title, content=content, author=author, date=date, file=file)

    # Convert the Post instance to a dictionary
    post_dict = post.__dict__

    # Convert the User instance in the 'author' field to a dictionary
    post_dict['author'] = post.author.__dict__

    # Add the post to Firestore
    db.collection('posts').add(post_dict)