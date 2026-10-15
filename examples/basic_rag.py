"""Run from the repository root: python examples/basic_rag.py document.pdf"""

import sys
from rag import RAGPipeline

pipeline = RAGPipeline()
pipeline.add_file(sys.argv[1])
result = pipeline.ask("What are the key points?")
print(result["response"])
