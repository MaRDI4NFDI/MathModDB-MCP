from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union, cast

from qdrant_client import QdrantClient, models

from embedder import (
    BatchDenseOutput,
    BatchMultivectorOutput,
    DenseOutput,
    MultivectorEmbedder,
    MultivectorOutput,
    TextEmbedder,
)
from graph import DEFAULT_EXCLUDED_PROPERTY_IDS
from models import Entity, MathModDBStructure
from result import (
    RetrievedClass,
    RetrievedDataProperty,
    RetrievedEntity,
    RetrievedObjectProperty,
    RetrievedQualifierProperty,
    SearchResult,
)

# Type alias for entities or strings to embed
EmbeddingInput = Union[Sequence[Entity | str], str]

# Collection names
CLASSES = "classes"
OBJECT_PROPERTIES = "object_properties"
DATA_PROPERTIES = "data_properties"
QUALIFIER_PROPERTIES = "qualifier_properties"

# Mapping from collection name to RetrievedEntity subclass
RETRIEVED_ENTITY_MAP = {
    CLASSES: RetrievedClass,
    OBJECT_PROPERTIES: RetrievedObjectProperty,
    DATA_PROPERTIES: RetrievedDataProperty,
    QUALIFIER_PROPERTIES: RetrievedQualifierProperty,
}


class QDrantStore:
    """Vector database store for ontology entities using QDrant.

    Manages one collection per entity kind and provides unified search.

    Attributes:
        collections: Dictionary mapping collection names to QDrantCollection instances
        client: Shared QDrantClient instance used by all collections
        db_path: Path to database storage or ":memory:" for in-memory storage
        ontology: MathModDB ontology containing entities to embed
        dense_embedder: Dense-vector embedder for retrieval.
        multivector_embedder: Optional late-interaction embedder.
    """

    collections: dict[str, QDrantCollection]
    client: QdrantClient
    db_path: Literal[":memory:"] | Path
    ontology: MathModDBStructure
    dense_embedder: TextEmbedder
    multivector_embedder: Optional[MultivectorEmbedder]

    def __init__(
        self,
        db_path: Literal[":memory:"] | Path,
        ontology: MathModDBStructure,
        dense_embedder: TextEmbedder,
        multivector_embedder: Optional[MultivectorEmbedder] = None,
    ):
        """Initialize QDrant store with ontology and embedder.

        Creates separate collections for classes, object properties, data
        properties, and qualifier properties.
        Uses a single shared QDrantClient instance to prevent concurrent access errors.

        Args:
            db_path: Storage path (use ":memory:" for in-memory storage)
            ontology: MathModDB ontology containing entities to store
            dense_embedder: Dense embedder used for base retrieval.
            multivector_embedder: Optional multivector embedder for reranking.
        """
        self.db_path = db_path
        self.ontology = ontology
        self.dense_embedder = dense_embedder
        self.multivector_embedder = multivector_embedder

        # Create single shared client instance
        if isinstance(db_path, str) and db_path == ":memory:":
            self.client = QdrantClient(path=db_path)
        else:
            self.client = QdrantClient(path=str(db_path))

        # Pass shared client to collections
        self.collections = {
            CLASSES: QDrantCollection(
                name=CLASSES,
                client=self.client,
                dense_embedder=dense_embedder,
                multivector_embedder=multivector_embedder,
                structure=ontology,
            ),
            OBJECT_PROPERTIES: QDrantCollection(
                name=OBJECT_PROPERTIES,
                client=self.client,
                dense_embedder=dense_embedder,
                multivector_embedder=multivector_embedder,
                structure=ontology,
            ),
            DATA_PROPERTIES: QDrantCollection(
                name=DATA_PROPERTIES,
                client=self.client,
                dense_embedder=dense_embedder,
                multivector_embedder=multivector_embedder,
                structure=ontology,
            ),
            QUALIFIER_PROPERTIES: QDrantCollection(
                name=QUALIFIER_PROPERTIES,
                client=self.client,
                dense_embedder=dense_embedder,
                multivector_embedder=multivector_embedder,
                structure=ontology,
            ),
        }

    def embed_ontology(self):
        """Embed all ontology entities into their respective collections.

        Processes classes, object properties, and data properties from the ontology
        and stores their vector embeddings in the corresponding QDrant collections.
        """
        self.collections[CLASSES].add_documents(
            self.ontology.classes,
        )
        self.collections[OBJECT_PROPERTIES].add_documents(
            self.ontology.object_properties,
        )
        self.collections[DATA_PROPERTIES].add_documents(
            self.ontology.data_properties,
        )
        self.collections[QUALIFIER_PROPERTIES].add_documents(
            self.ontology.qualifier_properties
        )

    def search_all(
        self,
        query: str,
        k: int = 5,
        score_threshold: float | None = None,
    ) -> SearchResult:
        """Search for entities across all collections and build a weighted graph.

        Performs vector similarity search across all entity collections.

        Args:
            query: Search query string to embed and match against
            k: Maximum number of results to return per collection.
            score_threshold: Minimum similarity score threshold (0.0-1.0).
                           If None, returns top k results regardless of score.

        Returns:
            Structured search result with retrieved entities and a weighted graph.
        """
        dense_vector = self.dense_embedder.embed(query)

        multivector = None
        if self.multivector_embedder is not None:
            multivector = self.multivector_embedder.embed(query)

        results = {}
        for name, collection in self.collections.items():
            results[name] = collection.search(
                dense_vector=dense_vector,
                multivector=multivector,
                score_threshold=score_threshold,
                k=k,
            )

        # Create a copy of the ontology graph
        graph = self.ontology.build_schema_graph(
            exclude=["qualifier"],
            excluded_property_ids=DEFAULT_EXCLUDED_PROPERTY_IDS,
        )

        obj_prop_scores = {
            prop.id: float(prop.score)
            for prop in results[OBJECT_PROPERTIES]
            if prop.id is not None
        }
        class_scores = {
            cls.id: float(cls.score) for cls in results[CLASSES] if cls.id is not None
        }
        data_prop_scores = {
            prop.id: float(prop.score)
            for prop in results[DATA_PROPERTIES]
            if prop.id is not None
        }
        qualifier_scores = {
            prop.id: float(prop.score)
            for prop in results[QUALIFIER_PROPERTIES]
            if prop.id is not None
        }

        def _percentile(values: list[float], q: float) -> float:
            if not values:
                return 0.0
            sorted_values = sorted(values)
            if len(sorted_values) == 1:
                return sorted_values[0]
            pos = (len(sorted_values) - 1) * q
            low = int(pos)
            high = min(low + 1, len(sorted_values) - 1)
            frac = pos - low
            return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac

        def _normalized_similarity_map(score_map: dict[str, float]) -> dict[str, float]:
            values = list(score_map.values())
            if not values:
                return {}
            lo = _percentile(values, 0.10)
            hi = _percentile(values, 0.90)
            spread = hi - lo
            if spread <= 1e-12:
                return {key: 0.5 for key in score_map}
            return {
                key: min(1.0, max(0.0, (score - lo) / spread))
                for key, score in score_map.items()
            }

        obj_prop_sim = _normalized_similarity_map(obj_prop_scores)
        class_sim = _normalized_similarity_map(class_scores)
        data_prop_sim = _normalized_similarity_map(data_prop_scores)
        qualifier_sim = _normalized_similarity_map(qualifier_scores)
        min_edge_weight = 0.05
        gamma = 1.5
        subject_similarity_weight = 0.2
        predicate_similarity_weight = 0.6
        object_similarity_weight = 0.2
        property_frequency_alpha = 0.35
        hub_penalty_alpha = 0.20

        property_frequency = Counter(
            edge.get("property_id")
            for edge in graph.edges()
            if edge.get("property_id") is not None
        )
        max_property_frequency = max(property_frequency.values(), default=1)

        node_degree = Counter[int]()
        for source_idx, target_idx, _ in graph.weighted_edge_list():
            node_degree[source_idx] += 1
            node_degree[target_idx] += 1
        max_node_degree = max(node_degree.values(), default=1)

        def _property_similarity(property_id: str | None) -> float:
            if property_id in obj_prop_sim:
                return obj_prop_sim[property_id]
            if property_id in qualifier_sim:
                return qualifier_sim[property_id]
            return 0.0

        def _node_similarity(node_payload: dict[str, Any]) -> float:
            node_kind = node_payload.get("kind")
            node_id = node_payload.get("id")
            if not isinstance(node_id, str):
                return 0.0
            if node_kind == "class":
                return class_sim.get(node_id, 0.0)
            if node_kind == "data_property":
                return data_prop_sim.get(node_id, 0.0)
            return 0.0

        def _similarity_cost(
            property_id: str | None,
            source_node: dict[str, Any],
            target_node: dict[str, Any],
        ) -> float:
            edge_similarity = (
                subject_similarity_weight * _node_similarity(source_node)
                + predicate_similarity_weight * _property_similarity(property_id)
                + object_similarity_weight * _node_similarity(target_node)
            )
            return min_edge_weight + (1.0 - edge_similarity) ** gamma

        def _frequency_penalty(property_id: str | None) -> float:
            if property_id is None or max_property_frequency <= 1:
                return 1.0
            frequency = property_frequency.get(property_id, 1)
            normalized = (frequency - 1) / (max_property_frequency - 1)
            return 1.0 + property_frequency_alpha * normalized

        for source_idx, target_idx, edge in graph.weighted_edge_list():
            base_weight = float(edge.get("weight", 1.0))
            prop_id = edge.get("property_id")
            source_node = graph.get_node_data(source_idx)
            target_node = graph.get_node_data(target_idx)
            degree_norm = (
                max(node_degree[source_idx], node_degree[target_idx]) / max_node_degree
            )
            hub_penalty = 1.0 + hub_penalty_alpha * degree_norm
            edge["weight"] = (
                base_weight
                * _similarity_cost(prop_id, source_node, target_node)
                * _frequency_penalty(prop_id)
                * hub_penalty
            )

        return SearchResult(
            classes=results[CLASSES],
            object_properties=results[OBJECT_PROPERTIES],
            data_properties=results[DATA_PROPERTIES],
            qualifier_properties=results[QUALIFIER_PROPERTIES],
            prefixes=self.ontology.prefixes,
            graph=graph,
        )


class QDrantCollection:
    """Individual QDrant collection for storing and searching entity embeddings.

    Manages a single collection within the QDrant vector database, handling
    embedding generation, document storage, and similarity search operations.

    Attributes:
        client: Shared QDrantClient instance for database operations
        embedder: TextEmbedder for generating vector representations
        name: Collection name identifier
        structure: MathModDBStructure reference to set on retrieved entities
    """

    client: QdrantClient
    dense_embedder: TextEmbedder
    multivector_embedder: Optional[MultivectorEmbedder]
    structure: Optional[MathModDBStructure]

    def __init__(
        self,
        name: str,
        client: QdrantClient,
        dense_embedder: TextEmbedder,
        multivector_embedder: Optional[MultivectorEmbedder] = None,
        structure: Optional[MathModDBStructure] = None,
    ):
        """Initialize a Qdrant collection wrapper.

        Args:
            name: Collection name.
            client: Shared Qdrant client.
            dense_embedder: Dense embedder used for vector storage/query.
            multivector_embedder: Optional multivector embedder.
            structure: Optional ontology reference for reconstructed entities.
        """
        self.name = name
        self.client = client
        self.dense_embedder = dense_embedder
        self.multivector_embedder = multivector_embedder
        self.structure = structure

        if not self.client.collection_exists(name):
            vector_params = {
                self.dense_embedder.name: self.dense_embedder.vector_params,
            }
            if self.multivector_embedder is not None:
                vector_params[self.multivector_embedder.name] = (
                    self.multivector_embedder.vector_params
                )

            self.client.create_collection(
                collection_name=name,
                vectors_config=vector_params,
            )

    def search(
        self,
        dense_vector: DenseOutput,
        multivector: Optional[MultivectorOutput] = None,
        k: int = 10,
        score_threshold: float | None = None,
    ) -> list[RetrievedEntity]:
        """Search the collection and return typed retrieved entities."""

        if self.multivector_embedder is not None and multivector is not None:
            results = self.client.query_points(
                collection_name=self.name,
                query=multivector,
                using=self.multivector_embedder.name,
                prefetch=[
                    models.Prefetch(
                        query=dense_vector,
                        using=self.dense_embedder.name,
                    )
                ],
                limit=k,
                score_threshold=score_threshold,
            )
        else:
            results = self.client.query_points(
                collection_name=self.name,
                query=dense_vector,
                using=self.dense_embedder.name,
                limit=k,
                score_threshold=score_threshold,
            )

        # Use the appropriate RetrievedEntity subclass based on collection name
        RetrievedType = RETRIEVED_ENTITY_MAP.get(self.name, RetrievedEntity)
        return [
            RetrievedType.from_qdrant_result(result, structure=self.structure)
            for result in results.points
        ]

    def add_documents(self, entities: EmbeddingInput):
        """Add entities to the collection with their embeddings.

        Generates embeddings for the provided entities and stores them
        in the QDrant collection for future similarity searches.

        Args:
            entities: List of Entity objects to embed and store

        Raises:
            AssertionError: If entities is not a list
            ValueError: If the upsert operation fails
        """
        assert isinstance(entities, list), (
            f"Invalid input type: {type(entities)} must be a list of Entities"
        )

        dense_embeddings = self._embed_dense(entities)

        if self.multivector_embedder is not None:
            multivector_embeddings = self._embed_multivector(entities)
        else:
            multivector_embeddings = None

        points = self._to_points(
            entities=cast(list[Entity], entities),
            dense_embeddings=dense_embeddings,
            multivector_embeddings=multivector_embeddings,
        )

        result = self.client.upsert(self.name, points)

        if result.status != models.UpdateStatus.COMPLETED:
            raise ValueError(f"Failed to add documents: {result.status}")

    def _to_points(
        self,
        entities: list[Entity],
        dense_embeddings: BatchDenseOutput,
        multivector_embeddings: Optional[BatchMultivectorOutput] = None,
    ):
        """Convert embeddings and entities to QDrant point structures.

        Args:
            entities: List of Entity objects corresponding to embeddings
            dense_embeddings: Dense embeddings for each entity.
            multivector_embeddings: Optional multivector embeddings.

        Returns:
            List of QDrant PointStruct objects ready for storage
        """
        points = []

        for idx, entity in enumerate(entities):
            vector: dict[str, Any] = {self.dense_embedder.name: dense_embeddings[idx]}
            if (
                self.multivector_embedder is not None
                and multivector_embeddings is not None
            ):
                vector[self.multivector_embedder.name] = multivector_embeddings[idx]

            point = models.PointStruct(
                id=idx,
                vector=vector,
                payload=entity.model_dump(),
            )
            points.append(point)

        return points

    def _embed_dense(self, entities: EmbeddingInput) -> BatchDenseOutput:
        """Generate embeddings using the configured embedder.

        Handles both string inputs and Entity objects, using the Entity's
        embed_text property for embedding generation when available.

        Args:
            entities: String, list of strings, or list of Entity objects to embed

        Returns:
            List of embedding vectors (list of floats)

        Raises:
            ValueError: If input type is not supported
        """
        if isinstance(entities, str):
            return self.dense_embedder.embed([entities])
        elif isinstance(entities, list):
            return self.dense_embedder.embed(
                [
                    entity.embed_text if isinstance(entity, Entity) else entity
                    for entity in entities
                ]
            )
        else:
            raise ValueError(f"Invalid input type: {type(entities)}")

    def _embed_multivector(self, entities: EmbeddingInput) -> BatchMultivectorOutput:
        """Generate embeddings using the configured multivector embedder.

        Handles both string inputs and Entity objects, using the Entity's
        embed_text property for embedding generation when available.

        Args:
            entities: String, list of strings, or list of Entity objects to embed
        """

        assert self.multivector_embedder is not None, (
            "Multivector embedder is not configured"
        )

        if isinstance(entities, str):
            return self.multivector_embedder.embed([entities])
        elif isinstance(entities, list):
            return self.multivector_embedder.embed(
                [
                    entity.embed_text if isinstance(entity, Entity) else entity
                    for entity in entities
                ]
            )
        else:
            raise ValueError(f"Invalid input type: {type(entities)}")
