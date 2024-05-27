from model import Vote
from dataclasses import asdict
import firebase_admin
from firebase_admin import firestore, credentials, initialize_app, storage

if not firebase_admin._apps:
    cred = credentials.Certificate('streamlit-ml-f44ba64799b5.json')
    firebase_admin.initialize_app(cred)

db = firestore.client()

def create_vote(user, post, value):
    vote = Vote(user=user, post=post, value=value)
    vote_dict = asdict(vote)  # Convert the Vote object to a dictionary
    vote_dict['user'] = asdict(vote.user)  # Convert the User object to a dictionary
    vote_dict['post'] = asdict(vote.post)  # Convert the Post object to a dictionary

    db.collection('votes').add(vote_dict)  # Add the vote to the 'votes' collection in Firestore

def count_vote(post_id):
    votes = db.collection('votes').where('post.id', '==', post_id).stream()
    count = 0
    for vote in votes:
        vote_dict = vote.to_dict()
        count += vote_dict['vote']
    return count