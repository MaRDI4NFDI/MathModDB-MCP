from __future__ import annotations

from functools import lru_cache
from typing import Any, List, Tuple

import rustworkx as rx
from pydantic import BaseModel
from toon_format import encode

from graph import DEFAULT_EXCLUDED_PROPERTY_IDS
from models import Entity, MathModDBStructure
from result import RetrievedEntity, SearchResult

WIKIBASE_DEFAULT_ENTITY_PREFIX = "wd"


class SubGraph(BaseModel):
    """Graph subset represented as entity tuples.

    Attributes:
        triples: Directed semantic triples as (subject, property, object).
        data_properties: Extractable data properties by class as (class, data_property).
        qualifiers: Qualifier usage triples as
            (subject_class, qualifier, qualified_property).
    """

    triples: List[Tuple[RetrievedEntity, RetrievedEntity, RetrievedEntity, int]]
    data_properties: List[Tuple[RetrievedEntity, RetrievedEntity]]
    qualifiers: List[Tuple[RetrievedEntity, RetrievedEntity, RetrievedEntity]]

    def toon(self) -> str:
        """Serialize the subgraph payload in TOON format."""
        # Keep deterministic ordering for stable output.
        self.triples.sort(key=lambda x: x[0].id)  # type: ignore
        self.data_properties.sort(key=lambda x: x[0].id)  # type: ignore
        self.qualifiers.sort(key=lambda x: x[0].id)  # type: ignore

        subgraph = [
            {
                "subject_class": str(subject),
                "predicate_property": str(predicate),
                "object_class": str(object),
                "occurrence_count": occurrence_count,
            }
            for subject, predicate, object, occurrence_count in self.triples
        ]

        data_properties: dict[str, list[str]] = {}
        for class_entity, data_prop_entity in self.data_properties:
            data_property_key = str(data_prop_entity)
            data_properties.setdefault(data_property_key, []).append(str(class_entity))

        qualifiers: dict[str, list[dict[str, str]]] = {}
        for subject, qualifier, qualified_property in self.qualifiers:
            qualifier_key = str(qualifier)
            qualifiers.setdefault(qualifier_key, []).append(
                {
                    "subject_class": str(subject),
                    "qualified_property": str(qualified_property),
                }
            )

        return encode(
            {
                "subgraph (candidate WHERE clauses)": subgraph,
                "data_properties (SELECT/projection fields)": data_properties,
                "qualifiers (scoped to subgraph above — see ranked qualifier candidates for full list)": qualifiers,
            }
        )


def steiner_from_results(results: SearchResult) -> SubGraph:
    """Build a Steiner-based `SubGraph` from a `SearchResult`.

    The function computes a Steiner tree over classes and data properties,
    validates tree edges against the directed result graph, and converts the
    resulting topology into the `SubGraph` pydantic schema.

    Args:
        results: Search result containing retrieved entities and the directed
            topology graph built from the ontology.

    Returns:
        A `SubGraph` instance with semantic triples, class-data-property pairs,
        and qualifier relationships.
    """
    undirected_graph = results.graph.to_undirected()
    id_to_index, key_to_index = _get_mapping(results.graph)
    terminal_nodes = _get_terminal_nodes(results, id_to_index)

    steiner_tree = rx.steiner_tree(
        undirected_graph,
        terminal_nodes,
        lambda edge: float(edge.get("weight", 1.0)),
    )

    entity_by_id = _build_entity_index(results)
    structure = _get_structure(results)
    validated_edges = _get_validated_directed_edges(
        steiner_tree=steiner_tree,
        graph=results.graph,
        key_to_index=key_to_index,
    )

    triples = _build_triples(
        validated_edges=validated_edges,
        entity_by_id=entity_by_id,
        structure=structure,
    )

    data_properties = _build_data_properties(
        steiner_tree=steiner_tree,
        results=results,
        entity_by_id=entity_by_id,
    )

    qualifiers = _build_qualifiers(
        steiner_tree=steiner_tree,
        results=results,
        entity_by_id=entity_by_id,
        structure=structure,
    )

    return SubGraph(
        triples=triples,
        data_properties=data_properties,
        qualifiers=qualifiers,
    )


def _get_mapping(
    graph: rx.PyDiGraph,
) -> tuple[dict[str, int], dict[str, int]]:
    """Build node lookup maps for graph IDs and keys.

    Args:
        graph: Directed graph where nodes carry payload fields such as `id`
            and `key`.

    Returns:
        Tuple of:
            - `id_to_index`: Maps entity IDs to node indices.
            - `key_to_index`: Maps internal node keys (`kind:id`) to indices.
    """
    id_to_index = {node.get("id"): index for index, node in enumerate(graph.nodes())}

    key_to_index = {
        node.get("key"): index
        for index, node in enumerate(graph.nodes())
        if node.get("key") is not None
    }

    return id_to_index, key_to_index


def _get_terminal_nodes(
    results: SearchResult,
    id_to_index: dict[str, int],
) -> list[int]:
    """Return terminal node indices used for Steiner computation.

    Terminals are all unique retrieved classes and retrieved data properties
    that can be mapped to graph node indices.

    Args:
        results: Search result containing retrieved classes and data properties.
        id_to_index: Graph lookup from entity ID to node index.

    Returns:
        Ordered unique list of terminal node indices.
    """
    terminal_ids = [
        entity.id
        for entity in [*results.classes, *results.data_properties]
        if entity.id is not None
    ]
    terminal_nodes = [
        id_to_index[entity_id]
        for entity_id in dict.fromkeys(terminal_ids)
        if entity_id in id_to_index
    ]

    return terminal_nodes


def _build_entity_index(results: SearchResult) -> dict[str, RetrievedEntity]:
    """Build an ID -> entity index for all retrieved entity kinds.

    Args:
        results: Search result containing retrieved classes, object properties,
            data properties, and qualifier properties.

    Returns:
        Mapping from entity ID to the retrieved `Entity` instance.
    """
    entity_by_id: dict[str, RetrievedEntity] = {}
    for entity in [
        *results.classes,
        *results.object_properties,
        *results.data_properties,
        *results.qualifier_properties,
    ]:
        if entity.id:
            entity_by_id[entity.id] = entity
    return entity_by_id


def _to_retrieved_entity(entity: Entity, score: float = 0.0) -> RetrievedEntity:
    """Create a `RetrievedEntity` from an ontology `Entity` fallback."""
    return RetrievedEntity(
        prefix=entity.prefix,
        id=entity.id,
        label=entity.label,
        description=entity.description,
        summary=entity.summary,
        structure=entity.structure,
        score=score,
    )


def _get_structure(results: SearchResult) -> MathModDBStructure:
    """Return the ontology structure reference available in search results."""
    if results.classes and results.classes[0].structure is not None:
        return results.classes[0].structure
    if results.object_properties and results.object_properties[0].structure is not None:
        return results.object_properties[0].structure
    if results.data_properties and results.data_properties[0].structure is not None:
        return results.data_properties[0].structure
    if (
        results.qualifier_properties
        and results.qualifier_properties[0].structure is not None
    ):
        return results.qualifier_properties[0].structure
    raise ValueError("No structure found in search results")


def _resolve_retrieved_entity(
    *,
    entity_id: str,
    entity_by_id: dict[str, RetrievedEntity],
    structure,
) -> RetrievedEntity | None:
    """Resolve an entity ID to a retrieved entity, with ontology fallback."""
    resolved = entity_by_id.get(entity_id)
    if resolved is not None:
        return resolved
    if structure is None:
        return None
    try:
        entity = structure[entity_id]
    except Exception:
        return None
    return _to_retrieved_entity(entity, score=0.0)


def _find_matching_directed_edge(
    graph: rx.PyDiGraph,
    source: int,
    target: int,
    expected: dict[str, Any],
) -> dict[str, Any] | None:
    """Find the best matching directed edge for an undirected Steiner edge.

    Matching priority:
        1) exact relation + property_id match
        2) property_id-only match
        3) lowest-weight candidate fallback

    Args:
        graph: Original directed graph from the search result.
        source: Source node index candidate.
        target: Target node index candidate.
        expected: Undirected edge payload from the Steiner tree.

    Returns:
        Best matching directed edge payload, or `None` if no candidates exist.
    """
    try:
        edge_data = graph.get_all_edge_data(source, target)
    except Exception:
        return None

    candidates = edge_data if isinstance(edge_data, list) else [edge_data]
    candidates = [edge for edge in candidates if isinstance(edge, dict)]
    if not candidates:
        return None

    expected_relation = expected.get("relation")
    expected_property_id = expected.get("property_id")
    for edge in candidates:
        if (
            edge.get("relation") == expected_relation
            and edge.get("property_id") == expected_property_id
        ):
            return edge
    for edge in candidates:
        if edge.get("property_id") == expected_property_id:
            return edge
    return min(candidates, key=lambda edge: float(edge.get("weight", 1.0)))


def _get_validated_directed_edges(
    *,
    steiner_tree: rx.PyGraph,
    graph: rx.PyDiGraph,
    key_to_index: dict[str, int],
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Recover directed semantics for undirected Steiner tree edges.

    Each returned tuple contains:
        - source node payload (directed)
        - target node payload (directed)
        - matched directed edge payload

    Args:
        steiner_tree: Undirected Steiner tree produced from the search graph.
        graph: Original directed graph used for edge reconciliation.
        key_to_index: Mapping of node key (`kind:id`) to directed graph index.

    Returns:
        List of directed edge tuples ready for semantic conversion.
    """
    validated_edges: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []

    for source, target, edge in steiner_tree.weighted_edge_list():
        source_node = steiner_tree.get_node_data(source)
        target_node = steiner_tree.get_node_data(target)

        source_idx = key_to_index.get(source_node.get("key"))
        target_idx = key_to_index.get(target_node.get("key"))
        if source_idx is None or target_idx is None:
            continue

        forward = _find_matching_directed_edge(graph, source_idx, target_idx, edge)
        if forward is not None:
            validated_edges.append((source_node, target_node, forward))
            continue

        backward = _find_matching_directed_edge(graph, target_idx, source_idx, edge)
        if backward is not None:
            validated_edges.append((target_node, source_node, backward))

    return validated_edges


def _build_triples(
    *,
    validated_edges: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
    entity_by_id: dict[str, RetrievedEntity],
    structure,
) -> list[tuple[RetrievedEntity, RetrievedEntity, RetrievedEntity, int]]:
    """Convert validated directed edges into semantic triples.

    Args:
        validated_edges: Directed edge tuples produced by
            `_get_validated_directed_edges`.
        entity_by_id: Mapping of entity IDs to retrieved `Entity` objects.
        structure: Optional ontology structure used as fallback for entities
            not present in retrieved search results.

    Returns:
        Deduplicated `(subject, predicate, object, occurrence_count)` triples in stable order.
    """
    triples: list[tuple[RetrievedEntity, RetrievedEntity, RetrievedEntity, int]] = []
    seen: set[tuple[str, str, str]] = set()

    for source_node, target_node, edge in validated_edges:
        source_id = source_node.get("id")
        target_id = target_node.get("id")
        property_id = edge.get("property_id")
        if not source_id or not target_id or not property_id:
            continue

        subject = _resolve_retrieved_entity(
            entity_id=source_id,
            entity_by_id=entity_by_id,
            structure=structure,
        )
        predicate = _resolve_retrieved_entity(
            entity_id=property_id,
            entity_by_id=entity_by_id,
            structure=structure,
        )
        obj = _resolve_retrieved_entity(
            entity_id=target_id,
            entity_by_id=entity_by_id,
            structure=structure,
        )
        if subject is None or predicate is None or obj is None:
            continue

        key = (source_id, property_id, target_id)
        if key in seen:
            continue
        seen.add(key)
        triples.append((subject, predicate, obj, int(edge.get("usage_count", 0))))

    return triples


def _build_data_properties(
    *,
    steiner_tree: rx.PyGraph,
    results: SearchResult,
    entity_by_id: dict[str, RetrievedEntity],
    excluded_property_ids: set[str] = DEFAULT_EXCLUDED_PROPERTY_IDS,
) -> list[tuple[RetrievedEntity, RetrievedEntity]]:
    """Collect class/data-property pairs reachable from Steiner classes.

    The class side is restricted to classes present in the Steiner tree.
    Data properties are taken from retrieved class metadata (`data_property_ids`)
    and resolved to retrieved entities.

    Args:
        steiner_tree: Undirected Steiner tree.
        results: Search result containing retrieved entities and class metadata.
        entity_by_id: Mapping of entity IDs to retrieved `Entity` objects.

    Returns:
        Deduplicated `(class_entity, data_property_entity)` pairs, sorted
        deterministically (primary key: descending data-property score).
    """
    retrieved_data_props = {
        prop.id: prop for prop in results.data_properties if prop.id is not None
    }

    steiner_class_ids = [
        node.get("id") for node in steiner_tree.nodes() if node.get("kind") == "class"
    ]
    steiner_class_ids = [
        class_id for class_id in dict.fromkeys(steiner_class_ids) if class_id
    ]

    structure = _get_structure(results)

    data_property_to_classes: dict[str, set[str]] = {}
    for class_id in steiner_class_ids:
        cls = None
        if structure is not None:
            try:
                cls = structure[class_id]
            except Exception:
                cls = None
        if cls is None:
            continue

        for prop_id in getattr(cls, "data_property_ids", []):
            if not prop_id:
                continue
            if prop_id in excluded_property_ids:
                continue
            if prop_id not in retrieved_data_props:
                continue
            if prop_id not in data_property_to_classes:
                data_property_to_classes[prop_id] = set()
            data_property_to_classes[prop_id].add(class_id)

    pairs: list[tuple[RetrievedEntity, RetrievedEntity]] = []
    for prop_id in sorted(
        data_property_to_classes,
        key=lambda pid: retrieved_data_props[pid].score,
        reverse=True,
    ):
        data_prop_entity = entity_by_id.get(prop_id)
        if data_prop_entity is None:
            continue

        for class_id in sorted(data_property_to_classes[prop_id]):
            class_entity = entity_by_id.get(class_id)
            if class_entity is None and structure is not None:
                try:
                    ontology_class = structure[class_id]
                except Exception:
                    continue
                class_entity = _to_retrieved_entity(ontology_class, score=0.0)
            if class_entity is None:
                continue
            pairs.append((class_entity, data_prop_entity))

    return pairs


def _build_qualifiers(
    *,
    steiner_tree: rx.PyGraph,
    results: SearchResult,
    entity_by_id: dict[str, RetrievedEntity],
    structure: MathModDBStructure,
    excluded_property_ids: set[str] = DEFAULT_EXCLUDED_PROPERTY_IDS,
) -> list[tuple[RetrievedEntity, RetrievedEntity, RetrievedEntity]]:
    """Collect qualifier patterns from ontology structure.

    Args:
        steiner_tree: Undirected Steiner tree used to constrain subject classes.
        results: Search result used to constrain qualified properties
            (data + object properties).
        entity_by_id: Mapping of entity IDs to retrieved `Entity` objects.
        structure: Ontology structure used to enumerate qualifier paths.

    Returns:
        Deduplicated qualifier patterns as
        `(subject_class, qualifier_property, qualified_data_property)`.
    """
    qualifiers: list[tuple[RetrievedEntity, RetrievedEntity, RetrievedEntity]] = []
    seen: set[tuple[str, str, str]] = set()

    steiner_subject_ids: set[str] = set()
    for node in steiner_tree.nodes():
        if node.get("kind") != "class":
            continue
        node_id = node.get("id")
        if node_id:
            steiner_subject_ids.add(node_id)
    retrieved_data_property_ids = {
        data_prop.id for data_prop in results.data_properties if data_prop.id
    }
    retrieved_object_property_ids = {
        object_prop.id for object_prop in results.object_properties if object_prop.id
    }
    retrieved_property_ids = retrieved_data_property_ids | retrieved_object_property_ids

    @lru_cache(maxsize=None)
    def resolve_cached(entity_id: str) -> RetrievedEntity | None:
        return _resolve_retrieved_entity(
            entity_id=entity_id,
            entity_by_id=entity_by_id,
            structure=structure,
        )

    for qualifier in structure.qualifier_properties:
        qualifier_id = qualifier.id
        if not qualifier_id:
            continue

        if qualifier_id in excluded_property_ids:
            continue

        qualifier_entity = resolve_cached(qualifier_id)
        if qualifier_entity is None:
            continue
        if qualifier_entity.prefix != WIKIBASE_DEFAULT_ENTITY_PREFIX:
            qualifier_entity = qualifier_entity.model_copy(
                update={"prefix": WIKIBASE_DEFAULT_ENTITY_PREFIX}
            )

        for qualifier_path in qualifier.qualifier_paths:
            subject_id = qualifier_path.subject_class_id
            qualified_property_id = qualifier_path.property_id

            if subject_id not in steiner_subject_ids:
                continue
            if qualified_property_id not in retrieved_property_ids:
                continue

            key = (subject_id, qualifier_id, qualified_property_id)
            if key in seen:
                continue
            seen.add(key)

            subject_entity = resolve_cached(subject_id)
            qualified_property_entity = resolve_cached(qualified_property_id)
            if subject_entity is None or qualified_property_entity is None:
                continue
            qualifiers.append(
                (subject_entity, qualifier_entity, qualified_property_entity)
            )

    return qualifiers
