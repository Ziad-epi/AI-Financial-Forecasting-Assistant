from __future__ import annotations

from dataclasses import dataclass
from typing import List

import chromadb


@dataclass
class VectorStoreConfig:
    persist_dir: str = "llm/chroma_db"
    collection_name: str = "financial_news"
    distance_space: str = "cosine"


class VectorStore:
    def __init__(self, config: VectorStoreConfig) -> None:
        self.config = config
        self.client = chromadb.PersistentClient(path=self.config.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": self.config.distance_space},
        )

    def reset(self) -> None:
        try:
            self.client.delete_collection(name=self.config.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": self.config.distance_space},
        )

    def add_texts(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
        ids: List[str],
    ) -> None:
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def count(self) -> int:
        return self.collection.count()

    def query(self, query_embedding: List[float], k: int) -> List[dict]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        hits = []
        for doc, meta, dist in zip(docs, metas, distances):
            if doc is None:
                continue
            similarity = 1.0 - dist if self.config.distance_space == "cosine" else None
            hits.append(
                {
                    "text": doc,
                    "metadata": meta or {},
                    "distance": dist,
                    "similarity": similarity,
                }
            )
        return hits


class Retriever:
    def __init__(self, vector_store: VectorStore, embedder) -> None:
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(self, query: str, k: int) -> List[dict]:
        query_embedding = self.embedder.embed_query(query)
        return self.vector_store.query(query_embedding=query_embedding, k=k)
