from dataclasses import dataclass
import datetime
from typing import List

@dataclass
class User:
    id: str
    name: str
    email: str

@dataclass
class Post:
    title: str
    content: str
    author: User
    datetime: datetime.datetime
    files: List[str]
    image: str = None

@dataclass
class Comment:
    content: str
    user: User
    datetime: datetime.datetime
    post: Post

@dataclass
class Vote:
    user: User
    post: Post
    value: int