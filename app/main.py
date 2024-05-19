from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
import models, schemas, crud
from database import SessionLocal, engine
from ml.text_search import TextSearch
from typing import List
import numpy as np

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

text_search = TextSearch()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    return crud.create_user(db=db, user=user)

@app.get("/users/", response_model=List[schemas.User])
def read_users(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    users = crud.get_users(db, skip=skip, limit=limit)
    return users

@app.put("/users/{user_id}", response_model=schemas.User)
def update_user(user_id: int, user: schemas.UserUpdate, db: Session = Depends(get_db)):
    db_user = crud.update_user(db, user_id, user)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@app.delete("/users/{user_id}", response_model=schemas.User)
def delete_user(user_id: int, db: Session = Depends(get_db)):
    db_user = crud.delete_user(db, user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@app.post("/records/", response_model=schemas.Record)
def create_record(record: schemas.RecordCreate, db: Session = Depends(get_db)):
    try:
        embedding = text_search.get_embedding(record.content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return crud.create_record(db=db, record=record, embedding=embedding)

@app.put("/records/{record_id}", response_model=schemas.Record)
def update_record(record_id: int, record: schemas.RecordUpdate, db: Session = Depends(get_db)):
    try:
        embedding = text_search.get_embedding(record.content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    db_record = crud.update_record(db, record_id, record, embedding)
    if db_record is None:
        raise HTTPException(status_code=404, detail="Record not found")
    return db_record

@app.get("/records/", response_model=List[schemas.Record])
def read_records(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    records = crud.get_records(db, skip=skip, limit=limit)
    return records


@app.delete("/records/{record_id}", response_model=schemas.Record)
def delete_record(record_id: int, db: Session = Depends(get_db)):
    db_record = crud.delete_record(db, record_id)
    if db_record is None:
        raise HTTPException(status_code=404, detail="Record not found")
    return db_record

@app.get("/similar_records/{record_id}", response_model=List[schemas.Record])
def find_similar_records(record_id: int, db: Session = Depends(get_db)):
    record = crud.get_record(db, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    
    records = crud.get_records(db)
    similarities = []
    for rec in records:
        if rec.id != record.id:
            similarity = text_search.cosine_similarity(np.array(record.embedding), np.array(rec.embedding))
            similarities.append((rec, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    similar_records = [rec[0] for rec in similarities[:5]]
    return similar_records


@app.post("/train_model/")
def train_word2vec_model(corpus: schemas.Corpus):
    text_search.train_model(corpus.corpus)
    return {"message": "Model trained and saved successfully."}
