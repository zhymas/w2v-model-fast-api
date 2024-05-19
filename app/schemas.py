from pydantic import BaseModel
from typing import List, Optional

class UserBase(BaseModel):
    name: str

class UserCreate(UserBase):
    pass

class UserUpdate(UserBase):
    pass

class User(UserBase):
    id: int

    class Config:
        from_attributes = True

class RecordBase(BaseModel):
    content: str
    user_id: int

class RecordCreate(RecordBase):
    pass

class RecordUpdate(RecordBase):
    pass

class Record(RecordBase):
    id: int
    embedding: Optional[List[float]] = None 

    class Config:
        orm_mode: True

class Corpus(BaseModel):
    corpus: List[List[str]]
    