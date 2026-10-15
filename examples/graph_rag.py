"""Run after installing the graph extra and spaCy model."""

import sys
from rag import RAGPipeline

pipeline = RAGPipeline(knowledge_graph=True)
pipeline.add_file(sys.argv[1])
result = pipeline.ask("What are the key concepts and their relationships?")
print(result["response"])
print(pipeline.knowledge_graph.statistics())
