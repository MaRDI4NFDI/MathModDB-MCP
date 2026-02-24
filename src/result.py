from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

import rustworkx as rx
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator
from qdrant_client import models
from toon_format import encode

from models import (
    Class,
    DataProperty,
    Entity,
    ObjectProperty,
    QualifierProperty,
)

if TYPE_CHECKING:
    from steiner import SubGraph

# Constants for model serialization include/exclude patterns
# These define which fields should be excluded when serializing SearchResult objects
SEARCH_RESULT_EXCLUDE: dict[str, Any] = {
    "classes": {
        "__all__": {
            "object_properties",
            "data_properties",
            "object_property_ids",
            "data_property_ids",
            "class_ids",
            "structure",
            "summary",
        },
    },
    "object_properties": {
        "__all__": {
            "classes",
            "class_ids",
            "structure",
            "common_connections",
            "summary",
        }
    },
    "data_properties": {
        "__all__": {
            "classes",
            "class_ids",
            "structure",
            "summary",
        }
    },
    "qualifier_properties": {
        "__all__": {
            "qualifier_paths",
            "summary",
        }
    },
}


class RetrievedEntity(Entity):
    """Entity retrieved from vector search with similarity score.

    Extends the base Entity class to include a similarity score from
    the vector database search results.

    Attributes:
        score: Similarity score from the vector search (0.0-1.0 for cosine similarity)
    """

    score: float

    @classmethod
    def from_qdrant_result(cls, result: models.ScoredPoint, structure=None):
        """Create RetrievedEntity from QDrant search result.

        Args:
            result: QDrant ScoredPoint containing entity data and similarity score
            structure: Optional MathModDBStructure reference to set on the entity

        Returns:
            RetrievedEntity instance with score and entity attributes

        Raises:
            AssertionError: If payload is None or not a dictionary
        """
        payload = result.payload
        assert payload is not None, "Payload is required"
        assert isinstance(payload, dict), "Payload must be a dictionary"

        entity = cls(
            score=result.score,
            **payload,
        )

        # Set structure reference if provided
        if structure is not None:
            entity.structure = structure

        return entity

    @property
    def embed_text(self) -> str:
        """`RetrievedEntity` is not intended for embedding generation."""
        raise NotImplementedError(
            "RetrievedEntity does not have an embed_text property"
        )


class RetrievedClass(RetrievedEntity, Class):
    """Class entity retrieved from vector search with similarity score.

    Inherits from both RetrievedEntity and Class, providing all Class
    attributes plus the similarity score.
    """

    object_property_ids: list[str] = Field(
        default_factory=list,
        exclude=True,
    )

    data_property_ids: list[str] = Field(
        default_factory=list,
        exclude=True,
    )

    @computed_field(alias=None)
    @cached_property
    def object_properties(self) -> list[ObjectProperty]:
        if self.structure is None:
            return []
        properties = []

        for object_property_id in self.object_property_ids:
            object_property = self.structure[object_property_id]

            if isinstance(object_property, ObjectProperty):
                properties.append(object_property)

        return properties

    @computed_field(alias=None)
    @cached_property
    def data_properties(self) -> list[DataProperty]:
        if self.structure is None:
            return []
        properties = []

        for data_property_id in self.data_property_ids:
            data_property = self.structure[data_property_id]

            if isinstance(data_property, DataProperty):
                properties.append(data_property)

        return properties


class RetrievedObjectProperty(RetrievedEntity, ObjectProperty):
    """Object property entity retrieved from vector search with similarity score.

    Inherits from both RetrievedEntity and ObjectProperty, providing all
    ObjectProperty attributes plus the similarity score.
    """

    # Exclude ID fields from serialization using native Pydantic Field
    class_ids: list[str] = Field(
        default_factory=list,
        exclude=True,
    )

    @computed_field(alias=None)
    @cached_property
    def classes(self) -> list[Class]:
        if self.structure is None:
            return []
        classes = []
        for class_id in self.class_ids:
            class_ = self.structure[class_id]
            if isinstance(class_, Class):
                classes.append(class_)
        return classes


class RetrievedDataProperty(RetrievedEntity, DataProperty):
    """Data property entity retrieved from vector search with similarity score.

    Inherits from both RetrievedEntity and DataProperty, providing all
    DataProperty attributes plus the similarity score.
    """

    class_ids: list[str] = Field(
        default_factory=list,
        exclude=True,
    )

    @computed_field(alias=None)
    @cached_property
    def classes(self) -> list[Class]:
        if self.structure is None:
            return []
        classes = []
        for class_id in self.class_ids:
            class_ = self.structure[class_id]
            if isinstance(class_, Class):
                classes.append(class_)
        return classes


class RetrievedQualifierProperty(RetrievedEntity, QualifierProperty):
    """Qualifier property entity retrieved from vector search with similarity score.

    Inherits from both RetrievedEntity and QualifierProperty, providing all
    QualifierProperty attributes plus the similarity score.
    """

    @field_validator("prefix", mode="after")
    @classmethod
    def _convert_prefix(cls, v: str) -> str:
        """Convert prefix to "pq"."""
        return "pq"


class SearchResult(BaseModel):
    """Search result from vector database.

    Represents the search results from vector database for all collections.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prefixes: dict[str, str] = Field(
        description="Prefixes for the ontology",
    )

    classes: list[RetrievedClass]
    object_properties: list[RetrievedObjectProperty]
    data_properties: list[RetrievedDataProperty]
    qualifier_properties: list[RetrievedQualifierProperty]
    graph: rx.PyDiGraph = Field(exclude=True)

    def toon(self) -> str:
        """Serialize SearchResult to TOON format.

        Uses SEARCH_RESULT_EXCLUDE to exclude computed fields, ID lists, and
        structure references, keeping only essential entity information
        (prefix, id, label, description, score).

        Returns:
            TOON-formatted string representation of the search results
        """
        result_dict = self.model_dump(exclude=SEARCH_RESULT_EXCLUDE, mode="json")
        toon_content = encode(result_dict)
        steiner_result = self.steiner()
        return toon_content + "\n\n" + steiner_result.toon()

    def steiner(self) -> SubGraph:
        """Build a Steiner-derived schema subgraph for this search result."""
        from steiner import steiner_from_results

        return steiner_from_results(self)
