"""Optional graph extraction and reranking, loaded only when requested."""

from .pipeline import VectorRetriever


class KnowledgeGraphExtension:
    """Expand retrieval using related entities, then rerank using graph connections."""

    def __init__(self, embedder, store, *, extractor=None, graph_ranking_weight=0.3):
        if not 0 <= graph_ranking_weight <= 1:
            raise ValueError("graph_ranking_weight must be between zero and one")
        from knowledge_graph.integration import KnowledgeGraphEnhancedRetrieval
        if extractor is None:
            from knowledge_graph.extractor import KnowledgeGraphExtractor
            extractor = KnowledgeGraphExtractor()
        self.extractor = extractor
        self.knowledge_graph = extractor.get_knowledge_graph()
        self.vector_retriever = VectorRetriever(embedder, store)
        self.ranker = KnowledgeGraphEnhancedRetrieval(
            store, self.knowledge_graph, graph_ranking_weight=graph_ranking_weight)

    def add_documents(self, texts, metadata):
        self.extractor.process_document_chunks(texts, [dict(item) for item in metadata])

    def search(self, query, top_k=5, filter_dict=None):
        entities = self.ranker.extract_entities_from_query(query)
        terms = self.ranker.generate_expansion_terms(entities)
        candidates = self.vector_retriever.search(query, top_k * 2, filter_dict)
        if terms:
            expanded = self.ranker.create_expanded_query(query, terms)
            additional = self.vector_retriever.search(expanded, top_k * 2, filter_dict)
            by_id = {doc['id']: doc for doc in candidates}
            for doc in additional:
                if doc['id'] not in by_id:
                    by_id[doc['id']] = doc
            candidates = list(by_id.values())
        return self.ranker.graph_rerank(candidates, entities)[:top_k]

    def clear(self):
        self.extractor.reset()
