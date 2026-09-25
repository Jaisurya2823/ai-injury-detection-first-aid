from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass(frozen=True)
class Evidence:
    document_id: str
    title: str
    source: str
    score: float
    text: str

class LocalRAG:
    def __init__(self, knowledge_path: str | Path):
        self.path=Path(knowledge_path)
        self.docs=json.loads(self.path.read_text(encoding="utf-8"))
        if not self.docs: raise ValueError("knowledge base is empty")
        corpus=[f"{d['title']} {d['text']}" for d in self.docs]
        self.vectorizer=TfidfVectorizer(stop_words="english", ngram_range=(1,2))
        self.matrix=self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int=3) -> list[Evidence]:
        query=(query or "").strip()
        if not query: return []
        vec=self.vectorizer.transform([query])
        scores=cosine_similarity(vec,self.matrix)[0]
        q_terms=set(query.lower().split())
        boosted=[]
        for i,d in enumerate(self.docs):
            title_terms=set(d["title"].lower().replace(":", " ").split())
            text_terms=set(d["text"].lower().replace("/", " ").split())
            overlap=len(q_terms & (title_terms | text_terms))
            title_overlap=len(q_terms & title_terms)
            boosted.append(float(scores[i]) + min(0.15, overlap*0.02) + min(0.25, title_overlap*0.08))
        order=np.argsort(np.asarray(boosted))[::-1][:max(1,top_k)]
        return [Evidence(self.docs[i]["document_id"],self.docs[i]["title"],self.docs[i]["source"],float(boosted[i]),self.docs[i]["text"]) for i in order if float(boosted[i])>0]
