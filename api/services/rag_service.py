from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from llm.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class RagMetadata:
    generator: str
    indexed: bool


class RagService:
    def __init__(self, generator_name: str) -> None:
        self.generator_name = generator_name
        self.pipeline: Optional[RAGPipeline] = None
        self.init_error: Optional[Exception] = None
        self.metadata: Optional[RagMetadata] = None

    def init(self) -> None:
        if self.pipeline is not None or self.init_error is not None:
            return

        try:
            logger.info("Initializing RAG pipeline (generator=%s)", self.generator_name)
            self.pipeline = RAGPipeline(generator_name=self.generator_name)

            indexed = False
            try:
                if self.pipeline.vector_store.count() == 0:
                    count = self.pipeline.index()
                    indexed = count > 0
                    logger.info("Indexed %s chunks into vector store", count)
                else:
                    indexed = True
            except Exception as exc:
                logger.warning("RAG indexing skipped: %s", exc)

            self.metadata = RagMetadata(generator=self.generator_name, indexed=indexed)
        except Exception as exc:
            self.init_error = exc
            logger.exception("Failed to initialize RAG pipeline")

    def ask(self, question: str) -> str:
        if self.pipeline is None and self.init_error is None:
            self.init()

        if self.init_error is not None:
            raise RuntimeError(f"RAG not available: {self.init_error}")
        if self.pipeline is None:
            raise RuntimeError("RAG pipeline not initialized")

        return self.pipeline.answer(question, debug=True)


def build_rag_service() -> RagService:
    generator = os.getenv("RAG_GENERATOR", "openai")
    return RagService(generator)
