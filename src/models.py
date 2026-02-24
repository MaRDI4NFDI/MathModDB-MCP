from __future__ import annotations

import asyncio
from abc import abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Union

import rustworkx as rx
from openai import AsyncOpenAI
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from graph import GraphPathResult, build_schema_graph, k_shortest_paths_between_ids
from llm import augment_description

MATHMODDB_WIKIBASE_ENDPOINT = "https://query.portal.mardi4nfdi.de/sparql"

console = Console()


class MathModDBStructure(BaseModel):
    """Represents the MathModDB ontology.

    Attributes:
        classes: Ontology classes.
        object_properties: Object properties connecting classes.
        data_properties: Data properties attached to classes.
        qualifier_properties: Qualifier properties that annotate statements.
    """

    prefixes: dict[str, str] = Field(
        default_factory=dict,
        description="Prefixes for the ontology (prefix -> namespace IRI)",
    )
    classes: list[Class] = []
    object_properties: list[ObjectProperty] = []
    data_properties: list[DataProperty] = []
    qualifier_properties: list[QualifierProperty] = []

    @cached_property
    def schema_graph(self) -> rx.PyDiGraph:
        """Build and cache the reusable schema graph representation."""
        return build_schema_graph(self)

    def build_schema_graph(
        self,
        excluded_property_ids: set[str] | None = None,
        relation_weights: Mapping[str, float] | None = None,
        exclude: list[
            Literal[
                "class",
                "object_property",
                "data_property",
                "qualifier",
            ]
        ]
        | None = None,
    ) -> rx.PyDiGraph:
        """Build a configurable schema graph for specific retrieval tasks.

        This is useful for experiments where you want to adjust excluded
        properties or edge weighting without invalidating ``schema_graph``.
        """
        return build_schema_graph(
            self,
            excluded_property_ids=excluded_property_ids,
            relation_weights=relation_weights,
            exclude=exclude,
        )

    def shortes_paths(
        self,
        source_id: str | Entity,
        target_id: str | Entity,
        *,
        source_kind: Literal["class", "object_property", "data_property", "qualifier"]
        | None = None,
        target_kind: Literal["class", "object_property", "data_property", "qualifier"]
        | None = None,
        k: int = 10,
        weighted: bool = True,
        max_hops: int | None = None,
        excluded_property_ids: set[str] | None = None,
        relation_weights: Mapping[str, float] | None = None,
        exclude: list[
            Literal[
                "class",
                "object_property",
                "data_property",
                "qualifier",
            ]
        ]
        | None = None,
    ) -> list[GraphPathResult]:
        """Return up to ``k`` shortest simple paths between two entity IDs.

        If no custom graph configuration is provided, this reuses cached
        ``schema_graph``. Otherwise, it builds a task-specific graph.
        """
        use_default_graph = (
            excluded_property_ids is None and relation_weights is None and not exclude
        )
        graph = (
            self.schema_graph
            if use_default_graph
            else self.build_schema_graph(
                excluded_property_ids=excluded_property_ids,
                relation_weights=relation_weights,
                exclude=exclude,
            )
        )

        if isinstance(source_id, Entity):
            assert source_id.id is not None
            source_id = source_id.id
            source_kind = self._infer_kind_from_id(source_id)
        if isinstance(target_id, Entity):
            assert target_id.id is not None
            target_id = target_id.id
            target_kind = self._infer_kind_from_id(target_id)

        # Infer kinds from canonical entity lookup when not explicitly provided.
        # This keeps API usage simple for common cases.
        if source_kind is None:
            source_kind = self._infer_kind_from_id(source_id)
        if target_kind is None:
            target_kind = self._infer_kind_from_id(target_id)

        return k_shortest_paths_between_ids(
            graph,
            source_id,
            target_id,
            source_kind=source_kind,
            target_kind=target_kind,
            k=k,
            weighted=weighted,
            max_hops=max_hops,
        )

    def _infer_kind_from_id(
        self,
        entity_id: str,
    ) -> Literal["class", "object_property", "data_property", "qualifier"] | None:
        """Infer graph node kind from entity ID via ``__getitem__``."""
        try:
            entity = self[entity_id]
        except KeyError:
            return None

        if isinstance(entity, Class):
            return "class"
        if isinstance(entity, ObjectProperty):
            return "object_property"
        if isinstance(entity, DataProperty):
            return "data_property"
        if isinstance(entity, QualifierProperty):
            return "qualifier"
        return None

    @cached_property
    def schema_graph_node_index_by_key(self) -> dict[str, int]:
        """Map typed schema node key (e.g. class:Q123) to graph index."""
        return {
            node_data["key"]: node_index
            for node_index, node_data in enumerate(self.schema_graph.nodes())
        }

    def schema_node_index(self, kind: str, entity_id: str) -> int | None:
        """Get cached node index for a typed entity key."""
        return self.schema_graph_node_index_by_key.get(f"{kind}:{entity_id}")

    def shortest_schema_path(
        self,
        source_key: str,
        target_key: str,
        *,
        weighted: bool = True,
        excluded_property_ids: set[str] | None = None,
        relation_weights: Mapping[str, float] | None = None,
    ) -> list[dict[str, Any]] | None:
        """Return the shortest path between two typed schema keys.

        Use cached ``schema_graph`` when using defaults; otherwise builds a
        task-specific graph with the provided exclusions/weights.
        """
        use_default_graph = excluded_property_ids is None and relation_weights is None
        graph = (
            self.schema_graph
            if use_default_graph
            else self.build_schema_graph(
                excluded_property_ids=excluded_property_ids,
                relation_weights=relation_weights,
            )
        )

        if use_default_graph:
            index_by_key = self.schema_graph_node_index_by_key
        else:
            index_by_key = {
                node_data["key"]: node_index
                for node_index, node_data in enumerate(graph.nodes())
            }

        source_index = index_by_key.get(source_key)
        target_index = index_by_key.get(target_key)
        if source_index is None or target_index is None:
            return None

        path_mapping = rx.dijkstra_shortest_paths(
            graph,
            source_index,
            weight_fn=(
                (lambda edge: float(edge.get("weight", 1.0)))
                if weighted
                else (lambda edge: 1.0)
            ),
        )
        try:
            path_indices = path_mapping[target_index]
        except KeyError:
            return None
        return [graph.get_node_data(node_index) for node_index in path_indices]

    def resolve_prefix(self, iri: str) -> str:
        """Resolve prefix from IRI, creating a new prefix if needed.

        Checks if the IRI starts with any existing prefix namespace.
        If found, returns that prefix. Otherwise, creates a new prefix
        "nsX" where X is an integer.

        Args:
            iri: Full IRI string

        Returns:
            Prefix string (e.g., "wd", "wdt", "ns1")
        """
        # Check if IRI matches any existing prefix namespace
        for prefix, namespace in self.prefixes.items():
            if iri.startswith(namespace):
                return prefix

        # No matching prefix found, create a new one
        # Find the next available nsX
        ns_num = 0
        while f"ns{ns_num}" in self.prefixes:
            ns_num += 1

        new_prefix = f"ns{ns_num}"

        if "/" in iri:
            namespace = iri.rsplit("/", 1)[0] + "/"
        else:
            namespace = iri + "/"

        self.prefixes[new_prefix] = namespace

        return new_prefix

    @model_validator(mode="after")
    def _set_structure_references(self):
        """Set structure reference on all entities in this structure."""
        for entity in (
            self.classes
            + self.object_properties
            + self.data_properties
            + self.qualifier_properties
        ):
            entity.structure = self
        return self

    @classmethod
    def from_wikibase(
        cls,
        endpoint: str = MATHMODDB_WIKIBASE_ENDPOINT,
        max_concurrent_requests: int = 2,
        sleep_seconds: float = 1.0,
        refresh: bool = False,
        prefixes: Union[dict[str, str], None] = None,
        cache_dir: str | Path | None = None,
    ) -> MathModDBStructure:
        """Create MathModDBStructure instance from WikiBase SPARQL endpoint.

        Results are cached in `.graph/` directory. Set `refresh=True` to ignore
        cache and fetch fresh data.

        Args:
            endpoint: SPARQL endpoint URL
            max_concurrent_requests: Maximum number of concurrent SPARQL requests
            sleep_seconds: Sleep duration between requests
            refresh: If True, ignore cache and fetch fresh data
            prefixes: Optional initial prefixes dict (prefix -> namespace IRI)
            cache_dir: Optional custom cache directory. Uses default temp cache when None.

        Returns:
            MathModDBStructure instance populated from WikiBase
        """
        from wikibase import initialize_kg_from_wikibase

        if prefixes is None:
            prefixes = {}

        return asyncio.run(
            initialize_kg_from_wikibase(
                endpoint=endpoint,
                max_concurrent_requests=max_concurrent_requests,
                sleep_seconds=sleep_seconds,
                refresh=refresh,
                initial_prefixes=prefixes,
                cache_dir=cache_dir,
            )
        )

    def enrich_descriptions(
        self,
        client: AsyncOpenAI,
        model: str = "gpt-4o-mini",
        max_concurrent_requests: int | None = 5,
    ) -> None:
        """Enrich entity descriptions with LLM summaries.

        Args:
            client: OpenAI client instance for API calls
            model: Model name to use for summaries
            max_concurrent_requests: Limit for concurrent LLM requests (None = no limit)
        """
        from wikibase import _get_cache_path, _save_to_cache

        asyncio.run(
            self._enrich_descriptions_async(
                client=client,
                model=model,
                max_concurrent_requests=max_concurrent_requests,
            )
        )

        cache_path = _get_cache_path(MATHMODDB_WIKIBASE_ENDPOINT)
        _save_to_cache(self, cache_path)

    async def _enrich_descriptions_async(
        self,
        client: AsyncOpenAI,
        model: str = "gpt-4o-mini",
        max_concurrent_requests: int | None = 5,
    ) -> None:
        if max_concurrent_requests is not None and max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requests must be >= 1 or None")

        semaphore = (
            asyncio.Semaphore(max_concurrent_requests)
            if max_concurrent_requests is not None
            else None
        )

        entities = self.classes + self.object_properties + self.data_properties
        if not entities:
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task(
                "Enriching descriptions",
                total=len(entities),
            )

            async def _enrich_entity(
                entity: Class | ObjectProperty | DataProperty,
            ) -> None:
                try:
                    entity.summary = await augment_description(
                        entity,
                        client,
                        model,
                        semaphore=semaphore,
                    )
                finally:
                    progress.update(task_id, advance=1)

            await asyncio.gather(*[_enrich_entity(entity) for entity in entities])

    def __getitem__(self, key: str) -> Entity:
        """Get an entity by ID."""
        for entity in (
            self.classes
            + self.object_properties
            + self.data_properties
            + self.qualifier_properties
        ):
            if entity.id == key:
                return entity
        raise KeyError(f"Entity {key} not found")

    def __contains__(self, key: str) -> bool:
        """Check whether an entity ID exists in the structure."""
        return any(
            entity.id == key
            for entity in (
                self.classes
                + self.object_properties
                + self.data_properties
                + self.qualifier_properties
            )
        )


class Entity(BaseModel):
    """Base class for ontology entities (classes, properties, etc.).

    Provides shared metadata fields and utility behavior for all entity types.

    Attributes:
        prefix: The prefix short form for the entity (e.g., "wd", "wdt", "ns1")
        id: The ID extracted from the IRI (e.g., Q1234567 or P123)
        label: Human-readable label for the entity
        description: Optional human-readable description
    """

    prefix: str = Field(
        ...,
        description="The prefix short form for the entity (e.g., 'wd', 'wdt', 'ns1')",
    )
    id: Optional[str] = Field(
        default=None,
        description="The ID extracted from the IRI (e.g., Q1234567 or P123)",
    )
    label: Optional[str] = Field(
        default=None,
        description="Human-readable label for the entity",
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional human-readable description",
    )
    summary: Optional[str] = Field(
        default=None,
        description="Optional summary of the entity generated by an LLM to be used for search.",
    )
    structure: Optional["MathModDBStructure"] = Field(
        default=None,
        exclude=True,
        description="Reference to the MathModDBStructure this entity belongs to.",
    )
    quantity: Optional[int] = Field(
        default=None,
        description="Quantity of the entity, e.g. number of instances, number of classes, etc.",
    )

    @property
    def iri(self) -> str:
        """Get the full IRI from prefix and id.

        Reconstructs the full IRI by combining the namespace from the prefix
        with the entity ID.

        Returns:
            Full IRI string

        Raises:
            ValueError: If structure is not set or prefix/id not found
        """
        if self.structure is None:
            raise ValueError("Entity structure reference not set")

        if self.prefix not in self.structure.prefixes:
            raise ValueError(f"Prefix '{self.prefix}' not found in structure prefixes")

        if self.id is None:
            raise ValueError("Entity ID is not set")

        namespace = self.structure.prefixes[self.prefix]
        return namespace + self.id

    @property
    @abstractmethod
    def embed_text(self) -> str:
        """Generate text representation for embedding/search purposes."""
        raise NotImplementedError("Subclasses must implement embed_text method")

    @field_validator("description", mode="before")
    @classmethod
    def _convert_description(cls, v):
        """Convert description to string or None.

        Args:
            v: Input value (string, list, or other)

        Returns:
            String description or None if empty

        Raises:
            ValueError: If input is not a string or list
        """
        if not v:
            return None

        if isinstance(v, list):
            return str(v[0])
        elif isinstance(v, str):
            return v
        else:
            raise ValueError(
                f"Invalid value for {cls.__name__}: {v}. Expected a list of strings or a string."
            )

    def __str__(self) -> str:
        prefix = self.prefix
        if isinstance(self, QualifierProperty):
            prefix = "pq"
        return f"{self.label} ({prefix}:{self.id})"


class Connection(BaseModel):
    """Represents a connection between two entities.

    Attributes:
        subject_id: The subject class ID (prefix stripped)
        object_id: The object class ID (prefix stripped)
    """

    subject_id: str = Field(..., description="The subject class ID")
    object_id: str = Field(..., description="The object class ID")
    usage_count: int = Field(
        default=0,
        description="Number of observed statements for this class-to-class connection",
    )


class QualifierPath(BaseModel):
    """Represents a subject/property usage path for a qualifier."""

    subject_class_id: str = Field(
        ...,
        description="The subject class ID where this qualifier is used",
    )
    property_id: str = Field(
        ...,
        description="The main property ID that this qualifier annotates",
    )
    usage_count: int = Field(
        default=0,
        description="Number of observed statements for this subject/property path",
    )


class ObjectProperty(Entity):
    """Represents an OWL object property.

    Object properties relate individuals to other individuals in the ontology.
    Inherits all attributes and methods from Entity.
    """

    class_ids: list[str] = Field(
        default_factory=list,
        description="List of class IDs that have this object property",
    )

    common_connections: list[Connection] = Field(
        default_factory=list,
        description="List of connections that have this object property",
    )

    @cached_property
    def embed_text(self) -> str:
        """Generate text representation for embedding/search purposes.

        Returns:
            Formatted string containing property name, description, domain, and range
        """
        if self.summary:
            summary = self.summary
        else:
            summary = self.description
        return "\n".join(
            [
                f"Label: {self.label}",
                f"Description: {summary}",
            ]
        )

    def get_relations(
        self,
        cls: Class,
        other_cls: Optional[Class] = None,
    ) -> list[Connection]:
        """Return class-to-class connections realized by this object property."""

        if other_cls:
            # We are looking for relations between two classes
            # governed by this object property
            # Ensure subject and object are different and match the two classes
            return [
                connection
                for connection in self.common_connections
                if (
                    connection.subject_id == cls.id
                    and connection.object_id == other_cls.id
                )
                or (
                    connection.subject_id == other_cls.id
                    and connection.object_id == cls.id
                )
            ]
        else:
            # We are looking for relations between this class and any other class
            # governed by this object property
            # Exclude self-links (where subject_id == object_id)
            return [
                connection
                for connection in self.common_connections
                if (connection.subject_id == cls.id or connection.object_id == cls.id)
                and connection.subject_id != connection.object_id
            ]


class QualifierProperty(Entity):
    """Represents an OWL qualifier property.

    Qualifier properties are used to qualify other properties.
    Inherits all attributes and methods from Entity.
    """

    qualifier_paths: list[QualifierPath] = Field(
        default_factory=list,
        description="Subject/property usage paths for this qualifier",
    )

    @cached_property
    def embed_text(self) -> str:
        """Generate text representation for embedding/search purposes."""
        if self.summary:
            summary = self.summary
        else:
            summary = self.description

        return "\n".join(
            [
                f"Label: {self.label}",
                f"Description: {summary}",
            ]
        )


class DataProperty(Entity):
    """Represents an OWL data property.

    Data properties relate individuals to literal values (strings, numbers, etc.).
    Inherits all attributes and methods from Entity.
    """

    class_ids: list[str] = Field(
        default_factory=list,
        description="List of class IDs that have this data property",
    )

    @cached_property
    def embed_text(self) -> str:
        """Generate text representation for embedding/search purposes.

        Returns:
            Formatted string containing property name, description, domain, and range
        """
        if self.summary:
            summary = self.summary
        else:
            summary = self.description

        return "\n".join(
            [
                f"Label: {self.label}",
                f"Description: {summary}",
            ]
        )


class Class(Entity):
    """Represents an OWL class.

    Classes define types of individuals in the ontology and can have
    hierarchical relationships with other classes.
    Inherits all attributes and methods from Entity.
    """

    object_property_ids: list[str] = Field(
        default_factory=list,
        description="List of object property IDs that have this class",
    )

    data_property_ids: list[str] = Field(
        default_factory=list,
        description="List of data property IDs that have this class",
    )

    @cached_property
    def embed_text(self) -> str:
        """Generate text representation for embedding/search purposes.

        Returns:
            Formatted string containing class name, description, and class IDs
        """
        if self.summary:
            summary = self.summary
        else:
            summary = self.description

        return "\n".join(
            [
                f"Label: {self.label}",
                f"Description: {summary}",
            ]
        )

    def connects(self, cls: Class) -> list[Connection]:
        """Return connections between this class and another class."""

        connections = []

        assert self.structure is not None, "Structure is not set"

        for object_property_id in self.object_property_ids:
            object_property = self.structure[object_property_id]

            if isinstance(object_property, ObjectProperty):
                connections.extend(object_property.get_relations(cls))

        return connections
