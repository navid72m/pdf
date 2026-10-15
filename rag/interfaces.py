"""Structural contracts for replaceable pipeline components."""

from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union


class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed(self, texts: Union[str, List[str]]) -> Any: ...


class Generator(Protocol):
    def generate_response(self, prompt: str, max_tokens: int = 512) -> str: ...


class DocumentStore(Protocol):
    def add_documents(self, documents: List[Any]) -> List[str]: ...

    def search(self, query_embedding: Any, top_k: int = 5,
               filter_func: Optional[Callable] = None) -> List[Tuple[Any, float]]: ...

    def count_documents(self) -> int: ...

    def clear(self) -> None: ...


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 5,
               filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]: ...


class GraphExtension(Retriever, Protocol):
    """Optional indexing and retrieval extension; instances belong to one pipeline."""

    def add_documents(self, texts: List[str], metadata: List[Dict[str, Any]]) -> None: ...

    def clear(self) -> None: ...
