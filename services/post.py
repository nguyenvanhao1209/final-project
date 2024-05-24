import firebase_admin
import os
from firebase_admin import firestore, credentials, initialize_app, storage
from model import Post, User

cred = credentials.Certificate("streamlit-ml-f44ba64799b5.json")

try:
    firebase_admin.get_app()
except ValueError as e:
    initialize_app(cred)

def create_post(title, content, author, datetime, files, image):
    db = firestore.client()
    bucket = storage.bucket('streamlit-ml.appspot.com')

    file_urls = []
    for file in files:
        file_name = file.name
        blob = bucket.blob(os.path.basename(file_name))

        blob.upload_from_file(file)

        blob.make_public()

        file_url = blob.public_url
        file_urls.append(file_url)

    # If image is not provided, set a default image URL
    if image is None:
        # Set your default image URL here
        image_url = "https://firebasestorage.googleapis.com/v0/b/streamlit-ml.appspot.com/o/dataset.jpg?alt=media&token=d0c8e436-6534-489f-8a59-eb02ea28c452"
    else:
        # Upload the provided image
        image_blob = bucket.blob(os.path.basename(image.name))
        image_blob.upload_from_file(image)
        image_blob.make_public()
        image_url = image_blob.public_url
    
    # Create a new Post instance
    post = Post(id=title, title=title, content=content, author=author, datetime=datetime, files=file_urls, image=image_url)

    # Convert the Post instance to a dictionary
    post_dict = post.__dict__

    # Convert the User instance in the 'author' field to a dictionary
    post_dict['author'] = post.author.__dict__

    # Add the post to Firestore
    db.collection('posts').add(post_dict)

def list_post():
    db = firestore.client()
    posts = db.collection('posts').get()
    post_list = []
    for post in posts:
        post_dict = post.to_dict()
        post_dict['id'] = post.id
        post_dict['author'] = User(**post_dict['author'])
        post_list.append(Post(**post_dict))
    return post_list