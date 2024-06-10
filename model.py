from dataclasses import dataclass
import datetime
from typing import List

@dataclass
class User:
    id: str
    name: str
    email: str
    image: str = None

@dataclass
class Post:
    id: str
    title: str
    content: str
    author: User
    datetime: datetime.datetime
    files: List[str]
    image: str = None
    downloaded: int = 0

@dataclass
class Comment:
    id: str
    content: str
    user: User
    datetime: datetime.datetime
    post: Post

@dataclass
class Vote:
    id: str
    user: User
    post: Post
    value: int