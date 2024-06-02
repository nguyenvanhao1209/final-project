from model import Vote
from dataclasses import asdict
import firebase_admin
from firebase_admin import firestore, credentials, initialize_app, storage
import uuid

if not firebase_admin._apps:
    cred = credentials.Certificate('streamlit-ml-f44ba64799b5.json')
    firebase_admin.initialize_app(cred)

db = firestore.client()

def create_vote(user, post, value):
    id = str(uuid.uuid4())
    vote = Vote(id=id, user=user, post=post, value=value)
    vote_dict = asdict(vote)  # Convert the Vote object to a dictionary
    vote_dict['user'] = asdict(vote.user)  # Convert the User object to a dictionary
    vote_dict['post'] = asdict(vote.post)  # Convert the Post object to a dictionary

    db.collection('votes').add(vote_dict)  # Add the vote to the 'votes' collection in Firestore

def is_voted(user_id, post_id):
    # Query the 'votes' collection for a vote with the given user ID and post ID
    votes = db.collection('votes').where('user.id', '==', user_id).where('post.id', '==', post_id).stream()

    for vote in votes:
        vote_dict = vote.to_dict()
        if vote_dict['value'] == 1:
            return True
    return False
    

def delete_vote(user_id, post_id):
    # Query the 'votes' collection for a vote with the given user ID and post ID
    votes = db.collection('votes').where('user.id', '==', user_id).where('post.id', '==', post_id).stream()

    for vote in votes:
        # Delete the vote
        db.collection('votes').document(vote.id).delete()

def count_vote(post_id):
    votes = db.collection('votes').where('post.id', '==', post_id).stream()
    count = 0
    for vote in votes:
        vote_dict = vote.to_dict()
        count += vote_dict['value']
    return count