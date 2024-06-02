from dataclasses import asdict
import firebase_admin
import os
from firebase_admin import firestore, credentials, initialize_app, storage
from model import Post, User
import uuid
from utils import get_file_extension
import requests

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
        image_url = "https://firebasestorage.googleapis.com/v0/b/streamlit-ml.appspot.com/o/dataset.png?alt=media&token=f63b17f7-c182-41c5-a65a-d36fee92d3f0"
    else:
        # Upload the provided image
        image_blob = bucket.blob(os.path.basename(image.name))
        image_blob.upload_from_file(image)
        image_blob.make_public()
        image_url = image_blob.public_url
    
    post_id = str(uuid.uuid4())
    # Create a new Post instance

    post = Post(id=post_id, title=title, content=content, author=author, datetime=datetime, files=file_urls, image=image_url, downloaded=0)

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

    post_list = sorted(post_list, key=lambda post: post.datetime, reverse=True)
    return post_list

def update_post(post_id, title, content, author, datetime, files, image):
    db = firestore.client()
    post_ref = db.collection('posts').document(post_id)

    update_data = {}
    if title is not None:
        update_data['title'] = title
    if content is not None:
        update_data['content'] = content
    if author is not None:
        update_data['author'] = asdict(author)
    if datetime is not None:
        update_data['datetime'] = datetime
    if files is not None:
        update_data['files'] = files
    if image is not None:
        update_data['image'] = image

    post_ref.update(update_data)

def delete_post(post_id):
    db = firestore.client()
    db.collection('posts').document(post_id).delete()

def search_post_by_title(title):
    posts = list_post()
    result = []
    for post in posts:
        if title.lower() in post.title.lower():
            result.append(post)
    return result

def search_post_by_file_type(posts, type_list):
    result = []
    for post in posts:
        for file in post.files:
            if get_file_extension(file) in type_list:
                result.append(post)
    return result

def search_post_by_file_size(posts, min_size, max_size):
    result = []
    for post in posts:
        file_sizes = 0
        for file in post.files:
            file_response = requests.head(file)
            file_size = file_response.headers.get(
                "Content-Length", "Unknown"
            )
            file_sizes += int(file_size)
        if min_size <= file_sizes <= max_size:
            result.append(post)
    return result

def update_downloaded(post_id):
    db = firestore.client()
    post_ref = db.collection('posts').document(post_id)
    post = post_ref.get()
    post_dict = post.to_dict()
    post_dict['downloaded'] += 1
    post_ref.update(post_dict)