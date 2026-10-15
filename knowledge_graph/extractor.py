"""
Enhanced knowledge graph module with improved entity extraction and reasoning capabilities.
"""

import os
import logging
import re
import networkx as nx
import spacy
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from collections import defaultdict
import uuid

# Configure logging
logger = logging.getLogger(__name__)


class EnhancedEntityExtractor:
    """
    Enhanced entity extraction with improved accuracy and additional extraction methods.
    """
    
    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        use_ner: bool = True,
        use_noun_phrases: bool = True,
        use_custom_rules: bool = True,
        min_entity_confidence: float = 0.5
    ):
        """
        Initialize the enhanced entity extractor.
        
        Args:
            spacy_model: spaCy model to use
            use_ner: Whether to use NER from spaCy
            use_noun_phrases: Whether to extract noun phrases
            use_custom_rules: Whether to use custom extraction rules
            min_entity_confidence: Minimum confidence threshold for entities
        """
        self.use_ner = use_ner
        self.use_noun_phrases = use_noun_phrases
        self.use_custom_rules = use_custom_rules
        self.min_entity_confidence = min_entity_confidence
        self.load_spacy(spacy_model)
        
        # Initialize custom patterns
        self.custom_patterns = []
        if use_custom_rules:
            self.initialize_custom_patterns()
    
    def load_spacy(self, model_name: str) -> None:
        """
        Load spaCy model.
        
        Args:
            model_name: Name of spaCy model to load
        """
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.warning(f"spaCy model {model_name} not found. Attempting to download...")
            try:
                # Try to download the model
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
                self.nlp = spacy.load(model_name)
                logger.info(f"Downloaded and loaded spaCy model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to download spaCy model: {e}")
                # Use simple tokenization as fallback
                self.nlp = spacy.blank("en")
                self.use_ner = False
    
    def initialize_custom_patterns(self) -> None:
        """Initialize custom entity extraction patterns."""
        # Add pattern matching for specific entity types
        patterns = []
        
        # Technology and product patterns
        tech_patterns = [
            {"label": "TECHNOLOGY", "pattern": [{"LOWER": "ai"}]},
            {"label": "TECHNOLOGY", "pattern": [{"LOWER": "artificial"}, {"LOWER": "intelligence"}]},
            {"label": "TECHNOLOGY", "pattern": [{"LOWER": "machine"}, {"LOWER": "learning"}]},
            {"label": "TECHNOLOGY", "pattern": [{"LOWER": "deep"}, {"LOWER": "learning"}]},
            {"label": "TECHNOLOGY", "pattern": [{"LOWER": "neural"}, {"LOWER": "network"}]},
            {"label": "TECHNOLOGY", "pattern": [{"LOWER": "nlp"}]},
            {"label": "TECHNOLOGY", "pattern": [{"LOWER": "natural"}, {"LOWER": "language"}, {"LOWER": "processing"}]}
        ]
        
        # Concept patterns
        concept_patterns = [
            {"label": "CONCEPT", "pattern": [{"LOWER": "rag"}]},
            {"label": "CONCEPT", "pattern": [{"LOWER": "retrieval"}, {"LOWER": "augmented"}, {"LOWER": "generation"}]},
            {"label": "CONCEPT", "pattern": [{"LOWER": "knowledge"}, {"LOWER": "graph"}]},
            {"label": "CONCEPT", "pattern": [{"LOWER": "vector"}, {"LOWER": "database"}]},
            {"label": "CONCEPT", "pattern": [{"LOWER": "embeddings"}]}
        ]
        
        # Combine all patterns
        patterns.extend(tech_patterns)
        patterns.extend(concept_patterns)
        
        # Add patterns to the NLP pipeline
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = (self.nlp.add_pipe("entity_ruler", before="ner")
                     if "ner" in self.nlp.pipe_names else self.nlp.add_pipe("entity_ruler"))
            ruler.add_patterns(patterns)
        else:
            ruler = self.nlp.get_pipe("entity_ruler")
            ruler.add_patterns(patterns)
    
    def extract_entities(self, text: str) -> List[Tuple[str, str, float]]:
        """
        Extract entities from text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of (entity name, entity type, confidence) tuples
        """
        entities = []
        
        if not text or not text.strip():
            return entities
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract named entities
        if self.use_ner:
            for ent in doc.ents:
                # Map spaCy entity types to our types
                entity_type = self._map_entity_type(ent.label_)
                
                # Calculate a confidence score (here we use a simple heuristic)
                confidence = min(1.0, 0.5 + 0.1 * len(ent.text.split()))
                
                if confidence >= self.min_entity_confidence:
                    entities.append((ent.text, entity_type, confidence))
        
        # Extract noun phrases as concept entities
        if self.use_noun_phrases and "parser" in self.nlp.pipe_names:
            for chunk in doc.noun_chunks:
                # Clean the noun phrase
                name = chunk.text.strip()
                
                # Skip short or non-informative phrases
                if len(name) < 3 or name.lower() in ["the", "a", "an", "this", "that", "these", "those"]:
                    continue
                
                # Give noun phrases a lower confidence
                confidence = 0.6
                
                # Check if it overlaps with a named entity (avoid duplicates)
                overlap = False
                for ent_name, _, _ in entities:
                    if ent_name in name or name in ent_name:
                        overlap = True
                        break
                
                if not overlap and confidence >= self.min_entity_confidence:
                    entities.append((name, "Concept", confidence))
        
        # Apply custom pattern matching for additional entities
        if self.use_custom_rules:
            self._apply_custom_patterns(text, entities)
        
        # Deduplicate entities
        unique_entities = []
        seen_names = set()
        
        for entity_name, entity_type, confidence in entities:
            key = (entity_name.lower(), entity_type)
            if key not in seen_names:
                seen_names.add(key)
                unique_entities.append((entity_name, entity_type, confidence))
        
        return unique_entities
    
    def _apply_custom_patterns(self, text: str, entities: List[Tuple[str, str, float]]) -> None:
        """
        Apply custom pattern matching for entity extraction.
        
        Args:
            text: Text to extract from
            entities: List to append extracted entities to
        """
        # Apply regex-based patterns
        patterns = [
            # URLs
            (r'https?://[^\s]+', 'URL'),
            # Email addresses
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'Email'),
            # File paths
            (r'(?:/[^/]+)+\.[a-zA-Z0-9]+', 'FilePath'),
            # Version numbers
            (r'\b\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9]+)?\b', 'Version')
        ]
        
        for pattern, entity_type in patterns:
            for match in re.finditer(pattern, text):
                entity_name = match.group(0)
                confidence = 0.8  # High confidence for pattern matches
                
                # Check if it's already been extracted
                exists = False
                for name, _, _ in entities:
                    if name == entity_name:
                        exists = True
                        break
                
                if not exists:
                    entities.append((entity_name, entity_type, confidence))
    
    def _map_entity_type(self, spacy_type: str) -> str:
        """
        Map spaCy entity types to our entity types.
        
        Args:
            spacy_type: spaCy entity type
            
        Returns:
            Mapped entity type
        """
        type_mapping = {
            "PERSON": "Person",
            "ORG": "Organization",
            "GPE": "Location",
            "LOC": "Location",
            "PRODUCT": "Product",
            "WORK_OF_ART": "WorkOfArt",
            "EVENT": "Event",
            "DATE": "Date",
            "TIME": "Time",
            "MONEY": "Money",
            "PERCENT": "Percent",
            "TECHNOLOGY": "Technology",
            "CONCEPT": "Concept"
        }
        
        return type_mapping.get(spacy_type, "Other")


class EnhancedRelationExtractor:
    """
    Enhanced relation extraction with improved accuracy and additional extraction methods.
    """
    
    def __init__(
        self,
        nlp,
        use_dependency_parsing: bool = True,
        use_pattern_matching: bool = True
    ):
        """
        Initialize the enhanced relation extractor.
        
        Args:
            nlp: spaCy NLP model
            use_dependency_parsing: Whether to use dependency parsing
            use_pattern_matching: Whether to use pattern matching
        """
        self.nlp = nlp
        self.use_dependency_parsing = use_dependency_parsing
        self.use_pattern_matching = use_pattern_matching
        
        # Initialize relation patterns
        self.relation_patterns = self._initialize_relation_patterns()
    
    def _initialize_relation_patterns(self) -> List[Dict[str, Any]]:
        """
        Initialize relation extraction patterns.
        
        Returns:
            List of pattern dictionaries
        """
        patterns = [
            {
                "name": "is_a_relation",
                "patterns": [
                    r"(?P<subject>[A-Z][a-z]+(?:\s+[A-Za-z]+)*) is (?:a|an) (?P<object>[a-z]+(?:\s+[a-z]+)*)",
                    r"(?P<subject>[A-Z][a-z]+(?:\s+[A-Za-z]+)*) are (?P<object>[a-z]+(?:\s+[a-z]+)*)"
                ],
                "relation": "is_a"
            },
            {
                "name": "part_of_relation",
                "patterns": [
                    r"(?P<subject>[A-Z][a-z]+(?:\s+[A-Za-z]+)*) is part of (?P<object>[A-Za-z]+(?:\s+[A-Za-z]+)*)",
                    r"(?P<subject>[A-Z][a-z]+(?:\s+[A-Za-z]+)*) belongs to (?P<object>[A-Za-z]+(?:\s+[A-Za-z]+)*)"
                ],
                "relation": "part_of"
            },
            {
                "name": "has_property",
                "patterns": [
                    r"(?P<subject>[A-Z][a-z]+(?:\s+[A-Za-z]+)*) has (?P<object>[a-z]+(?:\s+[a-z]+)*)",
                    r"(?P<subject>[A-Z][a-z]+(?:\s+[A-Za-z]+)*) contains (?P<object>[a-z]+(?:\s+[a-z]+)*)"
                ],
                "relation": "has_property"
            },
            {
                "name": "creates",
                "patterns": [
                    r"(?P<subject>[A-Z][a-z]+(?:\s+[A-Za-z]+)*) creates (?P<object>[A-Za-z]+(?:\s+[A-Za-z]+)*)",
                    r"(?P<subject>[A-Z][a-z]+(?:\s+[A-Za-z]+)*) produces (?P<object>[A-Za-z]+(?:\s+[A-Za-z]+)*)",
                    r"(?P<subject>[A-Z][a-z]+(?:\s+[A-Za-z]+)*) generates (?P<object>[A-Za-z]+(?:\s+[A-Za-z]+)*)"
                ],
                "relation": "creates"
            },
            {
                "name": "uses",
                "patterns": [
                    r"(?P<subject>[A-Z][a-z]+(?:\s+[A-Za-z]+)*) uses (?P<object>[A-Za-z]+(?:\s+[A-Za-z]+)*)",
                    r"(?P<subject>[A-Z][a-z]+(?:\s+[A-Za-z]+)*) utilizes (?P<object>[A-Za-z]+(?:\s+[A-Za-z]+)*)",
                    r"(?P<subject>[A-Z][a-z]+(?:\s+[A-Za-z]+)*) leverages (?P<object>[A-Za-z]+(?:\s+[A-Za-z]+)*)"
                ],
                "relation": "uses"
            }
        ]
        
        # Compile patterns
        for pattern in patterns:
            pattern["compiled_patterns"] = [re.compile(p) for p in pattern["patterns"]]
        
        return patterns
    
    def extract_relations(
        self,
        text: str,
        entities: List[Tuple[str, str, float]]
    ) -> List[Tuple[str, str, str, float]]:
        """
        Extract relations between entities.
        
        Args:
            text: Text to extract relations from
            entities: List of (entity_name, entity_type, confidence) tuples
            
        Returns:
            List of (subject, relation, object, confidence) tuples
        """
        relations = []
        
        if not text or not text.strip() or len(entities) < 2:
            return relations
        
        # Create entity mapping for easier lookup
        entity_map = {}
        for entity_name, entity_type, _ in entities:
            entity_map[entity_name] = entity_type
            # Also add lowercase version for matching
            entity_map[entity_name.lower()] = entity_type
        
        # Extract relations using pattern matching
        if self.use_pattern_matching:
            pattern_relations = self._extract_pattern_relations(text, entity_map)
            relations.extend(pattern_relations)
        
        # Extract relations using dependency parsing
        if self.use_dependency_parsing and "parser" in self.nlp.pipe_names:
            dependency_relations = self._extract_dependency_relations(text, entity_map)
            relations.extend(dependency_relations)
        
        # Deduplicate relations
        unique_relations = []
        seen_relations = set()
        
        for subj, rel, obj, conf in relations:
            key = (subj.lower(), rel, obj.lower())
            if key not in seen_relations:
                seen_relations.add(key)
                unique_relations.append((subj, rel, obj, conf))
        
        return unique_relations
    
    def _extract_pattern_relations(
        self,
        text: str,
        entity_map: Dict[str, str]
    ) -> List[Tuple[str, str, str, float]]:
        """
        Extract relations using pattern matching.
        
        Args:
            text: Text to extract from
            entity_map: Mapping of entity names to types
            
        Returns:
            List of (subject, relation, object, confidence) tuples
        """
        relations = []
        
        # Apply relation patterns
        for pattern in self.relation_patterns:
            for compiled_pattern in pattern["compiled_patterns"]:
                for match in compiled_pattern.finditer(text):
                    if "subject" in match.groupdict() and "object" in match.groupdict():
                        subj_text = match.group("subject")
                        obj_text = match.group("object")
                        
                        # Find matching entities
                        subj_entity = None
                        obj_entity = None
                        
                        # Direct match
                        if subj_text in entity_map:
                            subj_entity = subj_text
                        
                        if obj_text in entity_map:
                            obj_entity = obj_text
                        
                        # Fuzzy match if direct match fails
                        if subj_entity is None:
                            for entity_name in entity_map:
                                if (subj_text.lower() in entity_name.lower() or 
                                    entity_name.lower() in subj_text.lower()):
                                    subj_entity = entity_name
                                    break
                        
                        if obj_entity is None:
                            for entity_name in entity_map:
                                if (obj_text.lower() in entity_name.lower() or 
                                    entity_name.lower() in obj_text.lower()):
                                    obj_entity = entity_name
                                    break
                        
                        # If both entities found, add relation
                        if subj_entity and obj_entity:
                            # Calculate confidence (simplistic approach)
                            confidence = 0.7  # Pattern matches have moderate confidence
                            relations.append((subj_entity, pattern["relation"], obj_entity, confidence))
        
        return relations
    
    def _extract_dependency_relations(
        self,
        text: str,
        entity_map: Dict[str, str]
    ) -> List[Tuple[str, str, str, float]]:
        """
        Extract relations using dependency parsing.
        
        Args:
            text: Text to extract from
            entity_map: Mapping of entity names to types
            
        Returns:
            List of (subject, relation, object, confidence) tuples
        """
        relations = []
        
        # Parse text with spaCy
        doc = self.nlp(text)
        
        # Extract subject-verb-object relations
        for sent in doc.sents:
            for token in sent:
                # Find verbs that could be relation indicators
                if token.pos_ == "VERB":
                    # Extract the verb as relation
                    relation = token.lemma_.lower()
                    
                    # Find subject
                    subjects = []
                    for child in token.children:
                        if child.dep_ in ["nsubj", "nsubjpass"]:
                            # Include the full noun phrase if possible
                            start_idx = child.left_edge.i
                            end_idx = child.right_edge.i + 1
                            noun_phrase = doc[start_idx:end_idx].text
                            subjects.append(noun_phrase)
                    
                    # Find object
                    objects = []
                    for child in token.children:
                        if child.dep_ in ["dobj", "pobj"]:
                            # Include the full noun phrase if possible
                            start_idx = child.left_edge.i
                            end_idx = child.right_edge.i + 1
                            noun_phrase = doc[start_idx:end_idx].text
                            objects.append(noun_phrase)
                    
                    # Create relations between subjects and objects
                    for subj in subjects:
                        for obj in objects:
                            # Find matching entities
                            subj_entity = self._find_best_entity_match(subj, entity_map)
                            obj_entity = self._find_best_entity_match(obj, entity_map)
                            
                            # If both entities found, add relation
                            if subj_entity and obj_entity:
                                # Calculate confidence (simplistic approach)
                                confidence = 0.6  # Dependency parsing has lower confidence
                                relations.append((subj_entity, relation, obj_entity, confidence))
        
        return relations
    
    def _find_best_entity_match(self, phrase: str, entity_map: Dict[str, str]) -> Optional[str]:
        """
        Find the best matching entity in the entity map.
        
        Args:
            phrase: Phrase to match
            entity_map: Mapping of entity names to types
            
        Returns:
            Best matching entity name or None
        """
        # Try exact match
        if phrase in entity_map:
            return phrase
        
        # Try case-insensitive match
        phrase_lower = phrase.lower()
        if phrase_lower in entity_map:
            return phrase_lower
        
        # Try fuzzy matching
        best_match = None
        best_score = 0
        
        for entity_name in entity_map:
            # Calculate a simple overlap score
            if entity_name.lower() in phrase_lower:
                score = len(entity_name) / len(phrase)
                if score > best_score:
                    best_score = score
                    best_match = entity_name
            elif phrase_lower in entity_name.lower():
                score = len(phrase) / len(entity_name)
                if score > best_score:
                    best_score = score
                    best_match = entity_name
        
        # Only return if the match is good enough
        if best_score > 0.5:
            return best_match
        
        return None


class EnhancedKnowledgeGraph:
    """
    Enhanced knowledge graph implementation with improved capabilities.
    """
    
    def __init__(self):
        """Initialize the enhanced knowledge graph."""
        self.graph = nx.MultiDiGraph()  # Use MultiDiGraph to allow multiple edges between nodes
        self.entities = {}
        self.relations = {}
        
        # Track entity and relation frequencies for confidence
        self.entity_frequencies = defaultdict(int)
        self.relation_frequencies = defaultdict(int)
    
    def add_entity(
        self,
        name: str,
        entity_type: str,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add an entity to the knowledge graph.
        
        Args:
            name: Entity name
            entity_type: Entity type
            confidence: Confidence score (0.0-1.0)
            metadata: Optional metadata
            
        Returns:
            Entity ID
        """
        # Check if entity with same name and type already exists
        for entity_id, entity in self.entities.items():
            if entity["name"].lower() == name.lower() and entity["type"] == entity_type:
                # Update frequency count
                self.entity_frequencies[entity_id] += 1
                
                # Update confidence if new confidence is higher
                if confidence > entity["confidence"]:
                    entity["confidence"] = confidence
                
                # Update metadata if provided
                if metadata:
                    if "metadata" not in entity:
                        entity["metadata"] = {}
                    entity["metadata"].update(metadata)
                
                return entity_id
        
        # Create new entity
        entity_id = str(uuid.uuid4())
        
        # Add entity to graph
        self.graph.add_node(
            entity_id,
            name=name,
            type=entity_type,
            confidence=confidence
        )
        
        # Store entity
        self.entities[entity_id] = {
            "id": entity_id,
            "name": name,
            "type": entity_type,
            "confidence": confidence,
            "metadata": metadata or {}
        }
        
        # Update frequency
        self.entity_frequencies[entity_id] = 1
        
        logger.debug(f"Added entity: {name} ({entity_type})")
        return entity_id
    
    def add_relation(
        self,
        source: str,
        relation_type: str,
        target: str,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a relation between entities.
        
        Args:
            source: Source entity ID
            relation_type: Relation type
            target: Target entity ID
            confidence: Confidence score (0.0-1.0)
            metadata: Optional metadata
            
        Returns:
            Relation ID
        """
        # Verify entities exist
        if source not in self.entities:
            logger.warning(f"Source entity {source} not found. Skipping relation.")
            return None
        
        if target not in self.entities:
            logger.warning(f"Target entity {target} not found. Skipping relation.")
            return None
        
        # Check if relation already exists
        for rel_id, rel in self.relations.items():
            if (rel["source"] == source and 
                rel["target"] == target and 
                rel["type"] == relation_type):
                
                # Update frequency
                self.relation_frequencies[rel_id] += 1
                
                # Update confidence if new confidence is higher
                if confidence > rel["confidence"]:
                    rel["confidence"] = confidence
                
                # Update metadata if provided
                if metadata:
                    if "metadata" not in rel:
                        rel["metadata"] = {}
                    rel["metadata"].update(metadata)
                
                return rel_id
        
        # Create new relation
        relation_id = str(uuid.uuid4())
        
        # Add to graph
        self.graph.add_edge(
            source,
            target,
            key=relation_id,
            type=relation_type,
            confidence=confidence
        )
        
        # Store relation
        self.relations[relation_id] = {
            "id": relation_id,
            "source": source,
            "target": target,
            "type": relation_type,
            "confidence": confidence,
            "metadata": metadata or {}
        }
        
        # Update frequency
        self.relation_frequencies[relation_id] = 1
        
        logger.debug(f"Added relation: {self.entities[source]['name']} --[{relation_type}]--> {self.entities[target]['name']}")
        return relation_id
    
    def get_entity_by_name(self, name: str, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get entities by name.
        
        Args:
            name: Entity name to search for
            entity_type: Optional entity type filter
            
        Returns:
            List of matching entities
        """
        matches = []
        
        for entity_id, entity in self.entities.items():
            if entity["name"].lower() == name.lower():
                if entity_type is None or entity["type"] == entity_type:
                    matches.append(entity)
        
        return matches
    
    def get_neighbors(
        self,
        entity_id: str,
        relation_type: Optional[str] = None,
        direction: str = "outgoing"
    ) -> List[Dict[str, Any]]:
        """
        Get neighboring entities.
        
        Args:
            entity_id: Entity ID
            relation_type: Optional relation type filter
            direction: Direction of relations ('outgoing', 'incoming', 'both')
            
        Returns:
            List of neighboring entities with relation information
        """
        if entity_id not in self.entities:
            return []
        
        neighbors = []
        
        # Get outgoing relations
        if direction in ["outgoing", "both"]:
            for _, target, relation_id, edge_data in self.graph.out_edges(entity_id, keys=True, data=True):
                if relation_type is None or edge_data.get("type") == relation_type:
                    neighbors.append({
                        "entity": self.entities[target],
                        "relation": self.relations[relation_id],
                        "direction": "outgoing"
                    })
        
        # Get incoming relations
        if direction in ["incoming", "both"]:
            for source, _, relation_id, edge_data in self.graph.in_edges(entity_id, keys=True, data=True):
                if relation_type is None or edge_data.get("type") == relation_type:
                    neighbors.append({
                        "entity": self.entities[source],
                        "relation": self.relations[relation_id],
                        "direction": "incoming"
                    })
        
        return neighbors
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 3
    ) -> List[List[Dict[str, Any]]]:
        """
        Find paths between two entities.
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            max_hops: Maximum number of hops
            
        Returns:
            List of paths, where each path is a list of path steps
        """
        if source_id not in self.entities or target_id not in self.entities:
            return []
        
        try:
            # Find simple paths with NetworkX
            paths = list(nx.all_simple_paths(self.graph, source_id, target_id, cutoff=max_hops))
            
            # Format paths
            result_paths = []
            
            for path in paths:
                result_path = []
                
                # Add first entity
                result_path.append({
                    "entity": self.entities[path[0]],
                    "relation": None
                })
                
                # Add subsequent entities with their incoming relations
                for i in range(1, len(path)):
                    source = path[i-1]
                    target = path[i]
                    
                    # Find the specific edge data
                    edge_key = None
                    edges = self.graph.get_edge_data(source, target, default={})
                    if edges:
                        edge_key = max(edges, key=lambda key: self.relations[key]["confidence"])
                    
                    if edge_key:
                        result_path.append({
                            "entity": self.entities[target],
                            "relation": self.relations[edge_key]
                        })
                
                result_paths.append(result_path)
            
            return result_paths
            
        except nx.NetworkXNoPath:
            return []
    
    def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for entities by name, metadata, or text content.
        
        Args:
            query: Search query
            entity_type: Optional entity type filter
            min_confidence: Minimum confidence threshold
            limit: Maximum number of results
            
        Returns:
            List of matching entities
        """
        query = query.lower()
        matches = []
        
        for entity_id, entity in self.entities.items():
            # Skip if entity type doesn't match
            if entity_type and entity["type"] != entity_type:
                continue
                
            # Skip if confidence is too low
            if entity["confidence"] < min_confidence:
                continue
                
            # Check entity name
            if query in entity["name"].lower():
                matches.append(entity)
                continue
                
            # Check metadata
            if "metadata" in entity:
                for key, value in entity["metadata"].items():
                    if isinstance(value, str) and query in value.lower():
                        matches.append(entity)
                        break
        
        # Sort by confidence and frequency
        matches.sort(key=lambda e: (e["confidence"], self.entity_frequencies.get(e["id"], 0)), reverse=True)
        
        return matches[:limit]
    
    def suggest_related_entities(
        self,
        entity_id: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Suggest related entities based on graph structure.
        
        Args:
            entity_id: Entity ID to find related entities for
            max_results: Maximum number of results
            
        Returns:
            List of related entities with relevance scores
        """
        if entity_id not in self.entities:
            return []
        
        # Get direct neighbors
        neighbors = self.get_neighbors(entity_id, direction="both")
        neighbor_ids = set(n["entity"]["id"] for n in neighbors)
        
        # Find second-degree connections (neighbors of neighbors)
        second_degree = []
        for neighbor in neighbors:
            neighbor_id = neighbor["entity"]["id"]
            second_neighbors = self.get_neighbors(neighbor_id, direction="both")
            
            for second_neighbor in second_neighbors:
                second_id = second_neighbor["entity"]["id"]
                
                # Skip original entity and direct neighbors
                if second_id == entity_id or second_id in neighbor_ids:
                    continue
                
                # Calculate relevance score (simple heuristic)
                relevance = (
                    neighbor["relation"]["confidence"] *
                    second_neighbor["relation"]["confidence"] *
                    self.entity_frequencies.get(second_id, 1) / 10
                )
                
                second_degree.append({
                    "entity": self.entities[second_id],
                    "relevance": min(1.0, relevance)
                })
        
        # Sort by relevance
        second_degree.sort(key=lambda x: x["relevance"], reverse=True)
        
        return second_degree[:max_results]
    
    def get_entity_clusters(self, min_cluster_size: int = 3) -> List[List[Dict[str, Any]]]:
        """
        Find clusters of related entities in the knowledge graph.
        
        Args:
            min_cluster_size: Minimum size for a cluster
            
        Returns:
            List of entity clusters
        """
        # Convert to undirected graph for clustering
        undirected = self.graph.to_undirected()
        
        # Find connected components (clusters)
        components = list(nx.connected_components(undirected))
        
        # Filter and format clusters
        clusters = []
        
        for component in components:
            if len(component) >= min_cluster_size:
                cluster = []
                
                for entity_id in component:
                    if entity_id in self.entities:
                        cluster.append(self.entities[entity_id])
                
                if cluster:
                    clusters.append(cluster)
        
        # Sort clusters by size (largest first)
        clusters.sort(key=len, reverse=True)
        
        return clusters
    
    def calculate_centrality(self) -> Dict[str, float]:
        """
        Calculate centrality scores for all entities.
        
        Returns:
            Dictionary mapping entity IDs to centrality scores
        """
        try:
            # Calculate degree centrality
            centrality = nx.degree_centrality(self.graph)
            
            # Normalize scores to [0, 1]
            max_score = max(centrality.values()) if centrality else 1.0
            normalized = {k: v / max_score for k, v in centrality.items()}
            
            return normalized
        except Exception as e:
            logger.error(f"Error calculating centrality: {e}")
            return {}
    
    def find_key_entities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find key entities based on centrality and frequency.
        
        Args:
            limit: Maximum number of entities to return
            
        Returns:
            List of key entities with importance scores
        """
        # Calculate centrality
        centrality = self.calculate_centrality()
        
        # Combine centrality with frequency and confidence
        key_entities = []
        
        for entity_id, entity in self.entities.items():
            # Calculate importance score
            importance = (
                centrality.get(entity_id, 0) * 0.6 +  # Centrality
                min(1.0, self.entity_frequencies.get(entity_id, 0) / 5) * 0.3 +  # Frequency
                entity["confidence"] * 0.1  # Confidence
            )
            
            key_entities.append({
                "entity": entity,
                "importance": importance
            })
        
        # Sort by importance
        key_entities.sort(key=lambda x: x["importance"], reverse=True)
        
        return key_entities[:limit]
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the knowledge graph.
        
        Returns:
            Dictionary with summary information
        """
        # Calculate basic statistics
        entity_count = len(self.entities)
        relation_count = len(self.relations)
        
        # Count entity types
        entity_types = defaultdict(int)
        for entity in self.entities.values():
            entity_types[entity["type"]] += 1
        
        # Count relation types
        relation_types = defaultdict(int)
        for relation in self.relations.values():
            relation_types[relation["type"]] += 1
        
        # Find key entities
        key_entities = [item["entity"] for item in self.find_key_entities(5)]
        
        # Find key relationships
        key_relations = []
        for relation_id, relation in self.relations.items():
            # Include only if both entities are among key entities
            source_id = relation["source"]
            target_id = relation["target"]
            
            key_entity_ids = [e["id"] for e in key_entities]
            if source_id in key_entity_ids and target_id in key_entity_ids:
                key_relations.append({
                    "relation": relation,
                    "source": self.entities[source_id]["name"],
                    "target": self.entities[target_id]["name"]
                })
        
        # Find clusters
        clusters = self.get_entity_clusters()
        cluster_summaries = []
        
        for cluster in clusters[:3]:  # Top 3 clusters
            cluster_summaries.append({
                "size": len(cluster),
                "entities": [entity["name"] for entity in cluster[:5]]  # First 5 entities
            })
        
        return {
            "entity_count": entity_count,
            "relation_count": relation_count,
            "entity_types": dict(entity_types),
            "relation_types": dict(relation_types),
            "key_entities": key_entities,
            "key_relations": key_relations,
            "clusters": cluster_summaries
        }
    
    def clear(self) -> None:
        """Clear the knowledge graph."""
        self.graph = nx.MultiDiGraph()
        self.entities = {}
        self.relations = {}
        self.entity_frequencies = defaultdict(int)
        self.relation_frequencies = defaultdict(int)
        logger.info("Knowledge graph cleared")
    
    def save(self, file_path: str) -> None:
        """
        Save the knowledge graph to a file.
        
        Args:
            file_path: Path to save to
        """
        import json
        
        data = {
            "entities": self.entities,
            "relations": self.relations,
            "frequencies": {
                "entities": dict(self.entity_frequencies),
                "relations": dict(self.relation_frequencies)
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Knowledge graph saved to {file_path}")
    
    @classmethod
    def load(cls, file_path: str) -> 'EnhancedKnowledgeGraph':
        """
        Load a knowledge graph from a file.
        
        Args:
            file_path: Path to load from
            
        Returns:
            Loaded knowledge graph
        """
        import json
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        kg = cls()
        
        # Load entities
        for entity_id, entity in data["entities"].items():
            # Add to graph
            kg.graph.add_node(
                entity_id,
                name=entity["name"],
                type=entity["type"],
                confidence=entity.get("confidence", 1.0)
            )
            
            # Store entity
            kg.entities[entity_id] = entity
        
        # Load relations
        for relation_id, relation in data["relations"].items():
            # Add to graph
            source_id = relation["source"]
            target_id = relation["target"]
            
            kg.graph.add_edge(
                source_id,
                target_id,
                key=relation_id,
                type=relation["type"],
                confidence=relation.get("confidence", 1.0)
            )
            
            # Store relation
            kg.relations[relation_id] = relation
        
        # Load frequencies
        if "frequencies" in data:
            for entity_id, freq in data["frequencies"].get("entities", {}).items():
                kg.entity_frequencies[entity_id] = freq
            
            for relation_id, freq in data["frequencies"].get("relations", {}).items():
                kg.relation_frequencies[relation_id] = freq
        
        logger.info(f"Knowledge graph loaded from {file_path}")
        return kg
    def statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Dictionary of statistics
        """
        entity_types = {}
        for entity in self.entities.values():
            entity_types[entity["type"]] = entity_types.get(entity["type"], 0) + 1
        
        relation_types = {}
        for relation in self.relations.values():
            relation_types[relation["type"]] = relation_types.get(relation["type"], 0) + 1
        
        return {
            "num_entities": len(self.entities),
            "num_relations": len(self.relations),
            "entity_types": entity_types,
            "relation_types": relation_types,
            "is_connected": nx.is_weakly_connected(self.graph) if self.entities else True,
            "average_degree": sum(dict(self.graph.degree()).values()) / len(self.entities) if self.entities else 0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the knowledge graph to a dictionary.
        
        Returns:
            Dictionary representation of the knowledge graph
        """
        return {
            "entities": [dict(entity) for entity in self.entities.values()],
            "relations": [dict(relation) for relation in self.relations.values()]
        }
    


class KnowledgeGraphExtractor:
    """
    Enhanced knowledge graph extractor with improved entity and relation extraction.
    """
    
    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        use_ner: bool = True,
        use_noun_phrases: bool = True,
        use_custom_rules: bool = True,
        min_entity_confidence: float = 0.5
    ):
        """
        Initialize the enhanced knowledge graph extractor.
        
        Args:
            spacy_model: spaCy model to use
            use_ner: Whether to use NER for entity extraction
            use_noun_phrases: Whether to extract noun phrases as concepts
            use_custom_rules: Whether to use custom rules
            min_entity_confidence: Minimum confidence for entities
        """
        # Initialize entity extractor
        self.entity_extractor = EnhancedEntityExtractor(
            spacy_model=spacy_model,
            use_ner=use_ner,
            use_noun_phrases=use_noun_phrases,
            use_custom_rules=use_custom_rules,
            min_entity_confidence=min_entity_confidence
        )
        
        # Initialize relation extractor
        self.relation_extractor = EnhancedRelationExtractor(
            nlp=self.entity_extractor.nlp,
            use_dependency_parsing=True,
            use_pattern_matching=True
        )
        
        # Initialize knowledge graph
        self.kg= EnhancedKnowledgeGraph()
    
    def process_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EnhancedKnowledgeGraph:
        """
        Process text and extract knowledge graph.
        
        Args:
            text: Text to process
            metadata: Optional metadata
            
        Returns:
            The updated knowledge graph
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for processing")
            return self.kg
            
        logger.info(f"Processing text of length {len(text)}")
        
        # Extract entities
        entities = self.entity_extractor.extract_entities(text)
        logger.info(f"Extracted {len(entities)} entities")
        
        # Add entities to knowledge graph
        entity_map = {}  # Map entity names to IDs
        
        for entity_name, entity_type, confidence in entities:
            entity_id = self.kg.add_entity(
                name=entity_name,
                entity_type=entity_type,
                confidence=confidence,
                metadata=metadata
            )
            
            entity_map[entity_name] = entity_id
        
        # Extract relations
        relations = self.relation_extractor.extract_relations(text, entities)
        logger.info(f"Extracted {len(relations)} relations")
        
        # Add relations to knowledge graph
        for subject, relation_type, obj, confidence in relations:
            # Get entity IDs
            subject_id = entity_map.get(subject)
            object_id = entity_map.get(obj)
            
            if subject_id and object_id:
                self.kg.add_relation(
                    source=subject_id,
                    relation_type=relation_type,
                    target=object_id,
                    confidence=confidence,
                    metadata=metadata
                )
        
        return self.kg
    
    def process_document_chunks(
        self,
        chunks: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> EnhancedKnowledgeGraph:
        """
        Process document chunks and update the knowledge graph.
        
        Args:
            chunks: List of text chunks
            metadata_list: Optional list of metadata for each chunk
            
        Returns:
            The updated knowledge graph
        """
        if not chunks:
            logger.warning("No chunks provided for processing")
            return self.kg
            
        if metadata_list and len(metadata_list) != len(chunks):
            logger.warning(f"Metadata list length ({len(metadata_list)}) doesn't match chunks length ({len(chunks)})")
            metadata_list = None
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = None
            if metadata_list:
                chunk_metadata = metadata_list[i]
                # Add chunk index to metadata
                if chunk_metadata:
                    chunk_metadata["chunk_index"] = i
            
            self.process_text(chunk, chunk_metadata)
        
        logger.info(f"Processed {len(chunks)} document chunks")
        
        # Perform post-processing to enhance the knowledge graph
        self._post_process_graph()
        
        return self.kg
    
    def _post_process_graph(self) -> None:
        """
        Perform post-processing on the knowledge graph to improve quality.
        
        This includes removing low-confidence entities, inferring missing relations,
        and calculating centrality.
        """
        # Remove low-frequency, low-confidence entities
        entities_to_remove = []
        
        for entity_id, entity in self.kg.entities.items():
            frequency = self.kg.entity_frequencies.get(entity_id, 0)
            confidence = entity.get("confidence", 0.0)
            
            # Remove if both frequency and confidence are low
            if frequency == 1 and confidence < 0.6:
                entities_to_remove.append(entity_id)
        
        # Currently we don't have a remove_entity method in EnhancedKnowledgeGraph
        # but this would be implemented there
        
        # Infer missing relations
        self._infer_relations()
        
        logger.info("Completed knowledge graph post-processing")
    
    def _infer_relations(self) -> None:
        """
        Infer additional relations based on patterns in the existing graph.
        
        For example:
        - Transitive relations (A --is_a--> B --is_a--> C implies A --is_a--> C)
        - Inverse relations (A --part_of--> B implies B --has_part--> A)
        """
        # Define relation inference rules
        rules = [
            # Transitive is_a
            {
                "relation": "is_a",
                "transitive": True
            },
            # Inverse relations
            {
                "relation": "part_of",
                "inverse": "has_part"
            },
            {
                "relation": "uses",
                "inverse": "used_by"
            }
        ]
        
        # Apply transitive rules
        for rule in rules:
            if rule.get("transitive"):
                self._apply_transitive_rule(rule["relation"])
        
        # Apply inverse rules
        for rule in rules:
            if rule.get("inverse"):
                self._apply_inverse_rule(rule["relation"], rule["inverse"])
    
    def _apply_transitive_rule(self, relation_type: str) -> None:
        """
        Apply transitive relation rule.
        
        Args:
            relation_type: Type of relation to apply transitivity to
        """
        # Find all relations of the specified type
        transitive_relations = []
        
        for relation_id, relation in self.kg.relations.items():
            if relation["type"] == relation_type:
                source_id = relation["source"]
                target_id = relation["target"]
                confidence = relation.get("confidence", 0.5)
                
                transitive_relations.append((source_id, target_id, confidence))
        
        # Build adjacency map for faster lookup
        adjacency = defaultdict(list)
        for source, target, confidence in transitive_relations:
            adjacency[source].append((target, confidence))
        
        # Find transitive relations
        new_relations = []
        
        for source, target_list in adjacency.items():
            for target, confidence1 in target_list:
                # Look for next hop
                if target in adjacency:
                    for next_target, confidence2 in adjacency[target]:
                        # Avoid self-loops
                        if next_target != source:
                            # Calculate confidence for transitive relation
                            transitive_confidence = confidence1 * confidence2 * 0.9  # Decay factor
                            
                            # Check if relation already exists
                            exists = False
                            for relation in self.kg.relations.values():
                                if (relation["source"] == source and 
                                    relation["target"] == next_target and 
                                    relation["type"] == relation_type):
                                    exists = True
                                    break
                            
                            if not exists:
                                new_relations.append((source, relation_type, next_target, transitive_confidence))
        
        # Add inferred relations
        for source, rel_type, target, confidence in new_relations:
            self.kg.add_relation(
                source=source,
                relation_type=rel_type,
                target=target,
                confidence=confidence,
                metadata={"inferred": True, "rule": "transitive"}
            )
    
    def _apply_inverse_rule(self, relation_type: str, inverse_type: str) -> None:
        """
        Apply inverse relation rule.
        
        Args:
            relation_type: Original relation type
            inverse_type: Inverse relation type
        """
        # Find all relations of the specified type
        inverse_candidates = []
        
        for relation_id, relation in self.kg.relations.items():
            if relation["type"] == relation_type:
                source_id = relation["source"]
                target_id = relation["target"]
                confidence = relation.get("confidence", 0.5)
                
                # Check if inverse already exists
                exists = False
                for r in self.kg.relations.values():
                    if (r["source"] == target_id and 
                        r["target"] == source_id and 
                        r["type"] == inverse_type):
                        exists = True
                        break
                
                if not exists:
                    inverse_candidates.append((target_id, source_id, confidence))
        
        # Add inverse relations
        for source, target, confidence in inverse_candidates:
            self.kg.add_relation(
                source=source,
                relation_type=inverse_type,
                target=target,
                confidence=confidence * 0.95,  # Slightly lower confidence for inferred relations
                metadata={"inferred": True, "rule": "inverse"}
            )
    
    def get_knowledge_graph(self) -> EnhancedKnowledgeGraph:
        """
        Get the current knowledge graph.
        
        Returns:
            The knowledge graph
        """
        return self.kg
    
    def reset(self) -> None:
        """Reset the extractor and clear the knowledge graph."""
        self.kg.clear()
        logger.info("Knowledge graph extractor reset")
