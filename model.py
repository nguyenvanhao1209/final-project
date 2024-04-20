from dataclasses import dataclass

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
    date: str
    file: str