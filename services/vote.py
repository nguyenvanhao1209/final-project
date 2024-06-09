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
        if vote_dict['value']:
            return True
    return False
    

def delete_vote(user_id, post_id):
    # Query the 'votes' collection for a vote with the given user ID and post ID
    votes = db.collection('votes').where('user.id', '==', user_id).where('post.id', '==', post_id).stream()

    for vote in votes:
        # Delete the vote
        db.collection('votes').document(vote.id).delete()

def get_point_vote(post_id):
    votes = db.collection('votes').where('post.id', '==', post_id).stream()
    total_points = 0
    num_votes = 0
    for vote in votes:
        vote_dict = vote.to_dict()
        total_points += vote_dict['value']
        num_votes += 1
    if num_votes == 0:
        return 0.0
    else:
        return round(float(total_points / num_votes), 1)
    
def get_vote_user(user_id, post_id):
    votes = db.collection('votes').where('user.id', '==', user_id).where('post.id', '==', post_id).stream()

    for vote in votes:
        vote_dict = vote.to_dict()
        return vote_dict['value']

def change_vote(vote_id, new_value):
    vote = db.collection('votes').document(vote_id).get()
    vote_dict = vote.to_dict()
    vote_dict['value'] = new_value
    db.collection('votes').document(vote_id).update(vote_dict)

def get_vote_counts(post_id):
    votes = db.collection('votes').where('post.id', '==', post_id).stream()
    vote_counts = [0, 0, 0, 0, 0]  # Initialize vote counts for 5, 4, 3, 2, 1

    for vote in votes:
        vote_dict = vote.to_dict()
        vote_value = vote_dict['value']
        if vote_value in [1, 2, 3, 4, 5]:
            vote_counts[5 - vote_value] += 1  # Increment the count for the vote value

    return vote_counts

def get_total_votes(post_id):
    votes = db.collection('votes').where('post.id', '==', post_id).stream()
    total_votes = sum(1 for _ in votes)  # Count the number of votes
    return total_votes