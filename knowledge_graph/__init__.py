"""Knowledge graph components, imported lazily to keep integrations optional."""

from importlib import import_module

_EXPORTS = {
    "KnowledgeGraph": ".model", "Entity": ".model", "Relation": ".model",
    "KnowledgeGraphExtractor": ".extractor", "EnhancedKnowledgeGraph": ".extractor",
    "KnowledgeGraphEnhancedRetrieval": ".integration", "RAGEnhancer": ".integration",
    "KnowledgeGraphQueryProcessor": ".query",
    "generate_html_visualization": ".visualize", "get_graph_data": ".visualize",
    "save_graph_data_json": ".visualize",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
