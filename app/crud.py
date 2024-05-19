from sqlalchemy.orm import Session
import models, schemas
import numpy as np

def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()

def create_user(db: Session, user: schemas.UserCreate):
    db_user = models.User(name=user.name)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def update_user(db: Session, user_id: int, user: schemas.UserUpdate):
    db_user = get_user(db, user_id)
    if db_user is None:
        return None
    db_user.name = user.name
    db.commit()
    db.refresh(db_user)
    return db_user

def delete_user(db: Session, user_id: int):
    db_user = get_user(db, user_id)
    if db_user is None:
        return None
    db.delete(db_user)
    db.commit()
    return db_user

def get_record(db: Session, record_id: int):
    return db.query(models.Record).filter(models.Record.id == record_id).first()

def get_records(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Record).offset(skip).limit(limit).all()

def create_record(db: Session, record: schemas.RecordCreate, embedding: np.ndarray):
    db_record = models.Record(content=record.content, user_id=record.user_id, embedding=embedding)
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    return db_record

def update_record(db: Session, record_id: int, record: schemas.RecordUpdate, embedding: np.ndarray):
    db_record = get_record(db, record_id)
    if db_record is None:
        return None
    db_record.content = record.content
    db_record.embedding = embedding.tolist()
    db.commit()
    db.refresh(db_record)
    return db_record

def delete_record(db: Session, record_id: int):
    db_record = get_record(db, record_id)
    if db_record is None:
        return None
    db.delete(db_record)
    db.commit()
    return db_record
