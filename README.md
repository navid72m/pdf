# Document RAG Pipeline

A Python library for document ingestion, retrieval, and question answering. Compose a pipeline in a few lines, replace individual components, and optionally add knowledge graph retrieval.

## Install

Use Python 3.9 or newer. From the repository root:

```bash
pip install -e '.[rag,documents]'
```

The default embedding model downloads on first use. PDF and DOCX support comes from the `documents` extra; TXT and Markdown use Python's standard library.

## Create a pipeline

```python
from rag import RAGPipeline

rag = RAGPipeline()
rag.add_file("document.pdf")
result = rag.ask("What are the key points?")
print(result["response"])
```

Each pipeline owns its own document store. Defaults are SentenceTransformers embeddings, an in-memory FAISS index, and `LocalLLM`, an **extractive responder** that selects text from the retrieved chunks. Supply an LLM adapter for generative answers.

Add text directly or supply chunks you have already prepared:

```python
rag.add_text("Your document content...", metadata={"source": "notes"})
rag.add_documents(
    ["First prepared chunk", "Second prepared chunk"],
    metadata=[{"topic": "design"}, {"topic": "operations"}],
)

sources = rag.search("How does it work?", top_k=3)
result = rag.ask("Explain the design", filter_dict={"topic": "design"})
```

`ask()` returns a dictionary with `query`, `response`, `retrieved_documents`, and `knowledge_graph_used`. Each retrieved document contains `id`, `text`, `metadata`, and `score`. `count_documents()` counts chunks; `clear()` clears both vector and graph data.

## Optional knowledge graph

```bash
pip install -e '.[rag,documents,graph]'
python -m spacy download en_core_web_sm
```

```python
from rag import RAGPipeline

rag = RAGPipeline(knowledge_graph=True)
rag.add_file("document.pdf")
result = rag.ask("How are the main concepts related?")
print(result["response"])
print(rag.knowledge_graph.statistics())
```

The extension extracts entities and relationships at ingestion. During search it retrieves candidates for the original query and a query expanded with related entities, then reranks candidates using graph connections. Answer generation uses the retrieved document text. Extraction is based on English spaCy parsing and rules; inferred relationships are heuristics.

Standard pipelines do not import spaCy, NetworkX, Neo4j, or Streamlit. Graph extraction is opt-in. Install the spaCy model ahead of time; the existing extractor attempts a download when it cannot find the model.

For control over extraction and ranking:

```python
from rag import RAGPipeline
from rag.graph import KnowledgeGraphExtension
from embedding.model import create_embedding_model
from storage.vector_db import create_vector_database
from knowledge_graph.extractor import KnowledgeGraphExtractor

embedder = create_embedding_model()
store = create_vector_database(dimension=embedder.dimension)
graph = KnowledgeGraphExtension(
    embedder, store,
    extractor=KnowledgeGraphExtractor(min_entity_confidence=0.7),
    graph_ranking_weight=0.3,
)
rag = RAGPipeline(embedder=embedder, store=store, knowledge_graph=graph)
```

## Replace components

The constructor accepts `embedder`, `store`, `llm`, `processor`, `retriever`, `knowledge_graph`, `top_k`, and `prompt_template`. Components use structural protocols in `rag.interfaces`; inheritance is unnecessary.

| Component | Contract |
| --- | --- |
| Embedder | `dimension` property and `embed(text_or_texts)` returning one vector per input |
| Store | `add_documents(documents)`, `search(embedding, top_k, filter_func)`, `count_documents()`, `clear()` |
| LLM | `generate_response(prompt, max_tokens=512)` returning a string |
| Processor | `process_file(path, metadata=...)` and `process_text(text, metadata=...)`, returning chunks and matching metadata |
| Retriever | `search(query, top_k=5, filter_dict=None)` returning document dictionaries |
| Graph extension | Retriever contract plus `add_documents(texts, metadata)` and `clear()` |

For example, adapt an existing model client:

```python
from rag import RAGPipeline

class MyLLM:
    def __init__(self, client):
        self.client = client

    def generate_response(self, prompt, max_tokens=512):
        return self.client.complete(prompt=prompt, max_tokens=max_tokens)

# client is your application's model client, with a complete() method.
rag = RAGPipeline(llm=MyLLM(client))
```

A custom prompt must accept `{context}` and `{query}`. The default extractive responder also expects `Context:` and `Question:` headings. A custom retriever controls retrieval while the supplied store still handles ingestion; connect it to that same store. Supply either a custom retriever or a graph extension. An injected graph extension must likewise use the pipeline's embedder and store.

## Architecture

```text
RAGPipeline
  ├── Processor → chunks and metadata
  ├── Embedder → Store
  ├── Retriever → source chunks
  │     └── optional KnowledgeGraphExtension → expansion and reranking
  └── LLM → answer with source chunks
```

`rag/pipeline.py` owns orchestration. `rag/interfaces.py` defines component contracts. `rag/graph.py` provides the optional graph adapter. Existing document, embedding, vector store, and language model modules supply implementations.

The older `RAGEngine` and Streamlit/API integrations remain available through their existing modules. They have not been migrated to the new pipeline API; their legacy search-mode controls and graph UI are separate from this interface.

## Installation extras

| Extra | Purpose |
| --- | --- |
| `rag` | Default embedding and FAISS implementations |
| `documents` | PDF and DOCX extraction |
| `graph` | Entity extraction and graph retrieval |
| `neo4j` | Existing Neo4j persistence integration |
| `ui` | Streamlit and plotting dependencies |
| `api` | FastAPI, Uvicorn, and multipart uploads |

`pip install -e .` installs only the base dependencies for applications supplying their own components. `requirements.txt` retains the legacy application dependency list.

## Current limits

- Pipeline data is in memory unless the application manages persistence through its store. Passing a populated store does not automatically rebuild a graph.
- Vector insertion happens before graph extraction. A graph extraction error is raised, but already inserted chunks remain in the store. Ingestion is not transactional or deduplicated.
- Metadata filtering may scan the full vector index to preserve matching results.
- There is no conversation memory or token-budget management.
- Legacy Streamlit and API applications require further integration work before production use.

Runnable examples are in `examples/basic_rag.py` and `examples/graph_rag.py` after installing the project.
