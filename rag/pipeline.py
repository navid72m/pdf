"""Small, composable public API for document question answering."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .interfaces import DocumentStore, Embedder, Generator, GraphExtension, Retriever


DEFAULT_PROMPT = """Answer using only the provided context. If it is insufficient, say so.
Cite the document numbers supporting your answer.

Context:
{context}

Question: {query}

Answer:
"""


class VectorRetriever:
    """Adapt an embedding model and document store to the retrieval contract."""

    def __init__(self, embedder: Embedder, store: DocumentStore):
        self.embedder = embedder
        self.store = store

    def search(self, query: str, top_k: int = 5,
               filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        predicate = None
        if filter_dict:
            def predicate(document):
                for key, expected in filter_dict.items():
                    value = document.metadata
                    parts = key.split(".")
                    if parts[0] == "metadata":
                        parts = parts[1:]
                    for part in parts:
                        if not isinstance(value, dict) or part not in value:
                            return False
                        value = value[part]
                    if value != expected:
                        return False
                return True
        # Filtering after a limited FAISS search can miss valid matches.
        limit = self.store.count_documents() if predicate else top_k
        results = self.store.search(self.embedder.embed(query), limit, predicate)
        return [dict(id=doc.id, text=doc.text, metadata=doc.metadata, score=float(score))
                for doc, score in results[:top_k]]


class RAGPipeline:
    """Compose ingestion, retrieval and generation without a UI or server.

    Defaults use SentenceTransformers, FAISS and an extractive LocalLLM.
    Supply a Generator for model-backed answers. Components are owned per instance.
    """

    def __init__(self, *, embedder: Optional[Embedder] = None,
                 store: Optional[DocumentStore] = None,
                 llm: Optional[Generator] = None, processor=None,
                 retriever: Optional[Retriever] = None,
                 knowledge_graph: Union[bool, GraphExtension] = False,
                 top_k: int = 5, prompt_template: str = DEFAULT_PROMPT):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if retriever is not None and knowledge_graph is not False:
            raise ValueError("Supply either retriever or knowledge_graph")
        if embedder is None:
            from embedding.model import create_embedding_model
            embedder = create_embedding_model()
        if store is None:
            from storage.vector_db import create_vector_database
            store = create_vector_database(dimension=embedder.dimension)
        if llm is None:
            from llm.model import LocalLLM
            llm = LocalLLM()
        if processor is None:
            from document.processor import DocumentProcessor
            processor = DocumentProcessor()
        self.embedder, self.store, self.llm = embedder, store, llm
        self.processor = processor
        self.top_k = top_k
        self.prompt_template = prompt_template
        self.graph_extension = None
        if knowledge_graph is True:
            from .graph import KnowledgeGraphExtension
            self.graph_extension = KnowledgeGraphExtension(embedder, store)
        elif knowledge_graph is not False:
            self.graph_extension = knowledge_graph
        self.retriever = (self.graph_extension if self.graph_extension is not None
                          else retriever if retriever is not None
                          else VectorRetriever(embedder, store))

    @property
    def knowledge_graph(self):
        """Return the extension's graph when graph retrieval is enabled."""
        return getattr(self.graph_extension, "knowledge_graph", None)

    def add_file(self, path: Union[str, Path],
                 metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        chunks, metadata_list = self.processor.process_file(str(path), metadata=metadata)
        return self.add_documents(chunks, metadata_list)

    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        chunks, metadata_list = self.processor.process_text(text, metadata=metadata)
        return self.add_documents(chunks, metadata_list)

    def add_documents(self, texts: List[str],
                      metadata: Optional[List[Dict[str, Any]]] = None,
                      batch_size: int = 32) -> List[str]:
        """Index already chunked text. A graph error is raised after vector insertion."""
        from storage.vector_db import Document

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if metadata is not None and len(metadata) != len(texts):
            raise ValueError("Each text must have one metadata entry")
        metadata = [dict(item) for item in metadata] if metadata is not None else [{} for _ in texts]
        ids = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            embeddings = self.embedder.embed(batch)
            if len(embeddings) != len(batch):
                raise ValueError("Embedder must return one vector per text")
            documents = [Document(text=text, metadata=meta, embedding=embedding)
                         for text, meta, embedding in zip(
                             batch, metadata[start:start + batch_size], embeddings)]
            ids.extend(self.store.add_documents(documents))
        if self.graph_extension is not None and texts:
            self.graph_extension.add_documents(texts, metadata)
        return ids

    def search(self, query: str, *, top_k: Optional[int] = None,
               filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not query.strip():
            raise ValueError("query must not be empty")
        limit = self.top_k if top_k is None else top_k
        if limit < 1:
            raise ValueError("top_k must be positive")
        return self.retriever.search(query, top_k=limit, filter_dict=filter_dict)

    def ask(self, query: str, *, top_k: Optional[int] = None,
            filter_dict: Optional[Dict[str, Any]] = None,
            max_tokens: int = 512) -> Dict[str, Any]:
        """Return the answer and source chunks using the same shape in either mode."""
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        documents = self.search(query, top_k=top_k, filter_dict=filter_dict)
        if documents:
            context = "\n\n".join(
                f"Document {i}: [Source: {doc['metadata'].get('filename', doc['metadata'].get('source', 'Unknown'))}]\n{doc['text']}"
                for i, doc in enumerate(documents, 1))
            answer = self.llm.generate_response(
                self.prompt_template.format(context=context, query=query), max_tokens=max_tokens)
        else:
            answer = "I couldn't find any relevant information to answer your question."
        return {"query": query, "response": answer, "retrieved_documents": documents,
                "knowledge_graph_used": self.graph_extension is not None}

    def count_documents(self) -> int:
        """Count indexed chunks."""
        return self.store.count_documents()

    def clear(self) -> None:
        self.store.clear()
        if self.graph_extension is not None:
            self.graph_extension.clear()
