from gensim.models import Word2Vec
from typing import List
import numpy as np

class TextSearch:
    def __init__(self):
        self.model = None

    def train_model(self, corpus: List[List[str]]):
        self.model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)
        self.model.save("word2vec.model")

    def load_model(self):
        try:
            self.model = Word2Vec.load("word2vec.model")
        except FileNotFoundError:
            print("Word2Vec model file not found. You may need to train the model first.")

    def get_embedding(self, text: str) -> list:
        if self.model is None:
            raise ValueError("Model not loaded. Train the model first.")
        words = text.split()
        word_vectors = [self.model.wv[word] for word in words if word in self.model.wv]
        if word_vectors:
            return np.mean(word_vectors, axis=0).tolist()
        else:
            return np.zeros(self.model.vector_size).tolist()  


    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
