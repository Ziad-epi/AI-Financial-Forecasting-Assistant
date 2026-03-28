from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Tuple

from .embedding import EmbeddingConfig, EmbeddingModel
from .generator import build_generator
from .retriever import Retriever, VectorStore, VectorStoreConfig


@dataclass
class ChunkingConfig:
    chunk_size: int = 700
    chunk_overlap: int = 80


@dataclass
class PipelineConfig:
    data_path: str = "llm/data/financial_news.txt"
    persist_dir: str = "llm/chroma_db"
    collection_name: str = "financial_news"
    k: int = 4
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)


class Tokenizer:
    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.encoding_name = encoding_name
        self._tiktoken = None
        self._encoding = None

        try:
            import tiktoken

            self._tiktoken = tiktoken
            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            self._tiktoken = None
            self._encoding = None

    def encode(self, text: str) -> List[str] | List[int]:
        if self._encoding is None:
            return text.split()
        return self._encoding.encode(text)

    def decode(self, tokens: List[str] | List[int]) -> str:
        if self._encoding is None:
            return " ".join(tokens)  # type: ignore[arg-type]
        return self._encoding.decode(tokens)  # type: ignore[arg-type]


class RAGPipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        embed_config: EmbeddingConfig | None = None,
        generator_name: str = "openai",
    ) -> None:
        self.config = config or PipelineConfig()
        self.embedder = EmbeddingModel(embed_config)
        self.vector_store = VectorStore(
            VectorStoreConfig(
                persist_dir=self.config.persist_dir,
                collection_name=self.config.collection_name,
            )
        )
        self.retriever = Retriever(self.vector_store, self.embedder)
        self.generator = build_generator(generator_name)
        self.tokenizer = Tokenizer()

    def reset(self) -> None:
        self.vector_store.reset()

    def load_documents(self, data_path: str | None = None) -> List[str]:
        path = Path(data_path or self.config.data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        text = path.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return paragraphs

    def chunk_text(self, text: str) -> List[str]:
        tokens = self.tokenizer.encode(text)
        chunk_size = self.config.chunking.chunk_size
        overlap = self.config.chunking.chunk_overlap
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap >= chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")

        chunks = []
        start = 0
        total = len(tokens)
        while start < total:
            end = min(start + chunk_size, total)
            chunk_tokens = tokens[start:end]
            chunks.append(self.tokenizer.decode(chunk_tokens))
            if end == total:
                break
            start = end - overlap
        return chunks

    def chunk_documents(
        self,
        documents: Iterable[str],
        start_doc_id: int = 0,
        source_name: str | None = None,
    ) -> List[Tuple[str, dict]]:
        chunked = []
        source = source_name or Path(self.config.data_path).name
        for doc_id, doc in enumerate(documents, start=start_doc_id):
            chunks = self.chunk_text(doc)
            for chunk_id, chunk in enumerate(chunks):
                metadata = {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "source": source,
                }
                chunked.append((chunk, metadata))
        return chunked

    def index(self, data_path: str | None = None, reset: bool = False) -> int:
        if reset:
            self.reset()

        documents = self.load_documents(data_path)
        source_name = Path(data_path or self.config.data_path).name
        chunked = self.chunk_documents(documents, source_name=source_name)

        texts = [chunk for chunk, _ in chunked]
        metadatas = [meta for _, meta in chunked]
        ids = [f"doc{m['doc_id']}_chunk{m['chunk_id']}" for m in metadatas]

        embeddings = self.embedder.embed_documents(texts)
        self.vector_store.add_texts(texts, embeddings, metadatas, ids)
        return len(texts)

    def add_documents(self, new_docs: List[str], source_name: str | None = None) -> int:
        # Offset doc_ids to avoid collisions when adding dynamically
        start_doc_id = self.vector_store.count()
        chunked = self.chunk_documents(
            new_docs,
            start_doc_id=start_doc_id,
            source_name=source_name,
        )
        texts = [chunk for chunk, _ in chunked]
        metadatas = [meta for _, meta in chunked]
        ids = [f"dyn_doc{m['doc_id']}_chunk{m['chunk_id']}" for m in metadatas]

        embeddings = self.embedder.embed_documents(texts)
        self.vector_store.add_texts(texts, embeddings, metadatas, ids)
        return len(texts)

    def retrieve(self, query: str, k: int | None = None) -> List[dict]:
        k = k or self.config.k
        return self.retriever.retrieve(query=query, k=k)

    def format_context(self, hits: List[dict]) -> str:
        return "\n\n---\n\n".join(hit["text"] for hit in hits)

    def answer(self, query: str, k: int | None = None, debug: bool = True) -> str:
        hits = self.retrieve(query, k=k)

        if debug:
            self.log_retrieval(hits)

        context = self.format_context(hits)
        return self.generator.generate(context=context, question=query)

    def log_retrieval(self, hits: List[dict]) -> None:
        print("\n[Retrieval Results]")
        for idx, hit in enumerate(hits, start=1):
            sim = hit.get("similarity")
            dist = hit.get("distance")
            meta = hit.get("metadata", {})
            score = f"similarity={sim:.4f}" if sim is not None else f"distance={dist:.4f}"
            print(f"{idx}. {score} | doc_id={meta.get('doc_id')} chunk_id={meta.get('chunk_id')}")
            print(hit["text"])
            print("-")
