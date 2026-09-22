from pathlib import Path

from .chunking import chunk_documents
from .config import Settings
from .document_loader import load_document
from .embedding import embed_texts
from .generation import generate_answer
from .prompt import build_prompt
from .retrieval import retrieve
from .vector_store import VectorStore


INSUFFICIENT_EVIDENCE = (
    "I couldn't find enough evidence in the indexed documents to answer this question."
)


class RAGPipeline:
    def __init__(
        self,
        settings=None,
        embedding_function=embed_texts,
        generation_function=generate_answer,
    ):
        self.settings = settings or Settings()
        self.embedding_function = embedding_function
        self.generation_function = generation_function
        self.vector_store = VectorStore()

    def index_file(self, path, filename=None, source=None):
        pages = load_document(
            path,
            filename=filename,
            source=source or filename,
        )
        chunks = chunk_documents(
            pages,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        embeddings = self.embedding_function(
            [chunk["text"] for chunk in chunks],
            self.settings.embedding_model_name,
        )
        self.vector_store.add(chunks, embeddings)
        return {
            "document_id": chunks[0]["metadata"]["document_id"],
            "filename": filename or Path(path).name,
            "chunks_indexed": len(chunks),
        }

    def ask(self, question):
        evidence = retrieve(
            question,
            self.vector_store,
            top_k=self.settings.top_k,
            relevance_threshold=self.settings.relevance_threshold,
            embedding_function=self.embedding_function,
            embedding_model_name=self.settings.embedding_model_name,
        )
        if not evidence:
            return {"answer": INSUFFICIENT_EVIDENCE, "sources": []}

        prompt = build_prompt(question, evidence)
        answer = self.generation_function(
            prompt,
            model_name=self.settings.llm_model_name,
            max_new_tokens=self.settings.max_new_tokens,
        )
        return {"answer": answer, "sources": _sources_from(evidence)}


def _sources_from(evidence):
    return [
        {
            "evidence_number": index,
            "text": item.get("text", ""),
            "filename": item.get("filename"),
            "document_id": item.get("document_id"),
            "page": item.get("page"),
            "source": item.get("source"),
            "chunk_id": item.get("chunk_id"),
            "score": item.get("score"),
        }
        for index, item in enumerate(evidence, start=1)
    ]
