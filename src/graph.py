from __future__ import annotations

import heapq
from itertools import count
from typing import Any, Literal, Mapping, Sequence, TypeAlias, cast

import rustworkx as rx
from pydantic import BaseModel, Field


class GraphPathStep(BaseModel):
    """One step in a graph path (node or traversed property edge)."""

    kind: Literal["class", "object_property", "data_property", "qualifier", "property"]
    id: str | None = None
    label: str | None = None
    can_be_qualifier: bool = False
    relation: str | None = None
    weight: float | None = None


class GraphPathResult(BaseModel):
    """Typed return object for shortest path results."""

    source_key: str
    target_key: str
    path_length: int
    total_weight: float
    edges: list[GraphPathStep] = Field(default_factory=list)
    steps: list[GraphPathStep] = Field(default_factory=list)


GraphPathStepKind: TypeAlias = Literal[
    "class",
    "object_property",
    "data_property",
    "qualifier",
    "property",
]

# Trivial/high-frequency WikiBase properties that tend to collapse distances and
# are usually not helpful for schema-level semantic path analysis.
DEFAULT_EXCLUDED_PROPERTY_IDS: set[str] = {
    "P31",  # instance of
    "P36",  # subclass of
    "P37",  # part of
    "P1460",  # MaRDI profile type
    "P1495",  # community marker,
    "P146",  # series ordinal
    "P1569",  # order number
    "P401",  # media legend
    "P1640",  # local image
    "P1690",  # related URL
    "P1560",  # contains
}

# Baseline relation weights for weighted shortest paths. Lower = preferred.
DEFAULT_RELATION_WEIGHTS: dict[str, float] = {
    "object_property_connection": 1.0,
    "qualifier_object_property_connection": 1.3,
    "qualifier_usage_property": 1.7,
}

OBJECT_PROPERTY_CONNECTION = "object_property_connection"
QUALIFIER_OBJECT_PROPERTY_CONNECTION = "qualifier_object_property_connection"
QUALIFIER_USAGE_PROPERTY = "qualifier_usage_property"


def build_schema_graph(
    structure: Any,
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
    """Build a rustworkx graph over schema nodes with edge-carried properties.

    The structure is expected to expose:
      - classes, object_properties, data_properties, qualifier_properties
      - class.object_property_ids, class.data_property_ids
      - object_property.class_ids, object_property.common_connections
      - qualifier.property_ids, qualifier.qualifier_paths

    Node kinds:
      - class
      - data_property

    Edge payloads carry object-property and qualifier information so paths stay
    compact and can still be linearized as class/property/class (or
    class/property/data_property) sequences.

    Args:
        structure: MathModDB-like structure object with entity collections.
        excluded_property_ids: Property IDs to ignore while constructing topology.
            Defaults to ``DEFAULT_EXCLUDED_PROPERTY_IDS``.
        relation_weights: Optional mapping from relation label to edge weight.
            Defaults to ``DEFAULT_RELATION_WEIGHTS``.
    """
    excluded_ids = (
        DEFAULT_EXCLUDED_PROPERTY_IDS
        if excluded_property_ids is None
        else set(excluded_property_ids)
    )
    weights = (
        DEFAULT_RELATION_WEIGHTS if relation_weights is None else dict(relation_weights)
    )
    excluded_kinds = set(exclude or [])

    graph = rx.PyDiGraph()
    node_indices: dict[str, int] = {}
    seen_edges: set[tuple[Any, ...]] = set()
    object_props_by_id = {
        prop.id: prop for prop in structure.object_properties if prop.id is not None
    }
    data_prop_label_by_id = {
        prop.id: (prop.label or prop.id)
        for prop in structure.data_properties
        if prop.id is not None
    }
    qualifier_property_ids = {
        qualifier.id
        for qualifier in structure.qualifier_properties
        if qualifier.id is not None
    }

    def _node_key(kind: str, entity_id: str) -> str:
        return f"{kind}:{entity_id}"

    def _add_entity_node(kind: str, entity: Any) -> None:
        entity_id = getattr(entity, "id", None)
        if entity_id is None:
            return
        key = _node_key(kind, entity_id)
        if key in node_indices:
            return
        node_indices[key] = graph.add_node(
            {
                "key": key,
                "id": entity_id,
                "kind": kind,
                "label": getattr(entity, "label", None),
            }
        )

    def _add_edge(source_key: str, target_key: str, payload: dict[str, Any]) -> None:
        source_idx = node_indices.get(source_key)
        target_idx = node_indices.get(target_key)
        if source_idx is None or target_idx is None or source_idx == target_idx:
            return
        relation = payload.get("relation")
        payload["weight"] = float(weights.get(str(relation), 1.0))
        edge_key = (
            source_idx,
            target_idx,
            relation,
            tuple(sorted(payload.items())),
        )
        if edge_key in seen_edges:
            return
        seen_edges.add(edge_key)
        graph.add_edge(source_idx, target_idx, payload)

    def _property_payload(
        *,
        relation: str,
        property_id: str | None,
        property_label: str | None,
        qualifier_id: str | None = None,
        usage_count: int | None = None,
        qualified_property_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "relation": relation,
            "property_id": property_id,
            "property_label": property_label,
            "property_can_be_qualifier": (
                property_id in qualifier_property_ids
                if property_id is not None
                else False
            ),
        }
        if qualifier_id is not None:
            payload["qualifier_id"] = qualifier_id
        if usage_count is not None:
            payload["usage_count"] = usage_count
        if qualified_property_id is not None:
            payload["qualified_property_id"] = qualified_property_id
        return payload

    for cls in structure.classes:
        if "class" in excluded_kinds:
            break
        _add_entity_node("class", cls)
    for prop in structure.data_properties:
        if "data_property" in excluded_kinds:
            break
        if prop.id in excluded_ids:
            continue
        _add_entity_node("data_property", prop)

    for prop in structure.object_properties:
        if "object_property" in excluded_kinds:
            break
        if prop.id is None:
            continue
        if prop.id in excluded_ids:
            continue
        for connection in prop.common_connections:
            subject_key = _node_key("class", connection.subject_id)
            object_key = _node_key("class", connection.object_id)
            _add_edge(
                subject_key,
                object_key,
                _property_payload(
                    relation=OBJECT_PROPERTY_CONNECTION,
                    property_id=prop.id,
                    property_label=prop.label or prop.id,
                    usage_count=connection.usage_count,
                ),
            )

    for qualifier in structure.qualifier_properties:
        if "qualifier" in excluded_kinds:
            break
        if qualifier.id is None:
            continue
        if qualifier.id in excluded_ids:
            continue

        for path in qualifier.qualifier_paths:
            if path.property_id in excluded_ids:
                continue
            class_key = _node_key("class", path.subject_class_id)
            data_prop_label = data_prop_label_by_id.get(path.property_id)
            if data_prop_label is not None:
                _add_edge(
                    class_key,
                    _node_key("data_property", path.property_id),
                    _property_payload(
                        relation=QUALIFIER_USAGE_PROPERTY,
                        qualifier_id=qualifier.id,
                        property_id=qualifier.id,
                        property_label=qualifier.label or qualifier.id,
                        usage_count=path.usage_count,
                        qualified_property_id=path.property_id,
                    ),
                )

            if "object_property" in excluded_kinds:
                continue
            object_prop = object_props_by_id.get(path.property_id)
            if object_prop is None:
                continue

            for connection in object_prop.common_connections:
                if connection.subject_id != path.subject_class_id:
                    continue
                _add_edge(
                    class_key,
                    _node_key("class", connection.object_id),
                    _property_payload(
                        relation=QUALIFIER_OBJECT_PROPERTY_CONNECTION,
                        qualifier_id=qualifier.id,
                        property_id=qualifier.id,
                        property_label=qualifier.label or qualifier.id,
                        usage_count=path.usage_count,
                        qualified_property_id=path.property_id,
                    ),
                )

    return _prune_duplicate_connections(graph)


def _prune_duplicate_connections(graph: rx.PyDiGraph) -> rx.PyDiGraph:
    """Return a graph with duplicate semantic connections removed.

    Duplicates are edges that connect the same directed node pair and carry identical
    payload fields. This acts as a final safety pass after graph construction.
    Also removes nodes that have no edges.
    """
    # First pass: identify nodes that have edges
    nodes_with_edges: set[int] = set()
    for source_idx, target_idx, _ in graph.weighted_edge_list():
        nodes_with_edges.add(source_idx)
        nodes_with_edges.add(target_idx)

    # Second pass: build pruned graph with only connected nodes
    pruned_graph = rx.PyDiGraph()
    node_index_map: dict[int, int] = {}
    for node_index, node_data in enumerate(graph.nodes()):
        if node_index in nodes_with_edges:
            node_index_map[node_index] = pruned_graph.add_node(dict(node_data))

    seen_edge_keys: set[tuple[Any, ...]] = set()
    for source_idx, target_idx, payload in graph.weighted_edge_list():
        edge_key = (
            source_idx,
            target_idx,
            tuple(sorted(payload.items())),
        )
        if edge_key in seen_edge_keys:
            continue
        seen_edge_keys.add(edge_key)
        pruned_graph.add_edge(
            node_index_map[source_idx],
            node_index_map[target_idx],
            dict(payload),
        )

    return pruned_graph


def k_shortest_paths_between_ids(
    graph: rx.PyDiGraph,
    source_id: str,
    target_id: str,
    *,
    source_kind: Literal["class", "object_property", "data_property", "qualifier"]
    | None = None,
    target_kind: Literal["class", "object_property", "data_property", "qualifier"]
    | None = None,
    k: int = 10,
    weighted: bool = True,
    max_hops: int | None = None,
) -> list[GraphPathResult]:
    """Return up to ``k`` shortest simple paths between two entity IDs.

    IDs may be ambiguous across node kinds (for example ``P31``); pass
    ``source_kind`` and ``target_kind`` to disambiguate.

    Returned paths contain:
      - ``steps``: alternating node / traversed-property / node sequence
      - ``edges``: traversed property-edge steps only (object/data/qualifier usage)
    """
    if k < 1:
        return []

    source_indices = _node_indices_for_id(graph, source_id, kind=source_kind)
    target_indices = set(_node_indices_for_id(graph, target_id, kind=target_kind))
    if not source_indices or not target_indices:
        return []

    if max_hops is None:
        max_hops = graph.num_nodes() - 1

    if k == 1:
        best_path = _best_dijkstra_path(
            graph,
            source_indices,
            target_indices,
            weighted=weighted,
            max_hops=max_hops,
        )
        if best_path:
            return [
                _path_result(
                    graph,
                    best_path,
                    total_weight=_path_total_cost(graph, best_path, weighted=weighted),
                )
            ]

    results: list[GraphPathResult] = []
    seen_complete_paths: set[tuple[int, ...]] = set()
    heap_counter = count()
    frontier: list[tuple[float, int, int, list[int]]] = []

    for source_index in source_indices:
        heapq.heappush(frontier, (0.0, 0, next(heap_counter), [source_index]))

    while frontier and len(results) < k:
        path_cost, hops, _, path = heapq.heappop(frontier)
        current = path[-1]

        if current in target_indices and len(path) > 1:
            path_key = tuple(path)
            if path_key in seen_complete_paths:
                continue
            seen_complete_paths.add(path_key)
            results.append(_path_result(graph, path, total_weight=path_cost))
            continue

        if hops >= max_hops:
            continue

        for neighbor in graph.neighbors(current):
            if neighbor in path:
                continue
            edge_step_cost = _edge_cost(graph, current, neighbor, weighted=weighted)
            heapq.heappush(
                frontier,
                (
                    path_cost + edge_step_cost,
                    hops + 1,
                    next(heap_counter),
                    path + [neighbor],
                ),
            )

    return results


def _node_indices_for_id(
    graph: rx.PyDiGraph,
    entity_id: str,
    kind: Literal["class", "object_property", "data_property", "qualifier"]
    | None = None,
) -> list[int]:
    """Return node indices matching an entity ID and optional node kind."""
    indices: list[int] = []
    for node_index, node_data in enumerate(graph.nodes()):
        if node_data.get("id") != entity_id:
            continue
        if kind is not None and node_data.get("kind") != kind:
            continue
        indices.append(node_index)
    return indices


def _best_edge_payload(graph: rx.PyDiGraph, source: int, target: int) -> dict[str, Any]:
    """Return the lightest edge payload between two nodes."""
    try:
        edge_data = graph.get_all_edge_data(source, target)
        if isinstance(edge_data, list) and edge_data:
            return min(edge_data, key=lambda edge: float(edge.get("weight", 1.0)))
        if isinstance(edge_data, dict):
            return edge_data
    except Exception:
        pass
    return {"relation": "unknown", "weight": 1.0}


def _edge_cost(graph: rx.PyDiGraph, source: int, target: int, weighted: bool) -> float:
    """Return traversal cost for one edge step."""
    if not weighted:
        return 1.0
    edge_payload = _best_edge_payload(graph, source, target)
    return float(edge_payload.get("weight", 1.0))


def _path_total_cost(graph: rx.PyDiGraph, path: Sequence[int], weighted: bool) -> float:
    """Compute accumulated edge cost for a full node path."""
    if len(path) < 2:
        return 0.0
    return sum(
        _edge_cost(graph, path[i], path[i + 1], weighted=weighted)
        for i in range(len(path) - 1)
    )


def _path_result(
    graph: rx.PyDiGraph,
    path: Sequence[int],
    *,
    total_weight: float,
) -> GraphPathResult:
    """Build a typed path result from an index path and cost."""
    node_payloads, edge_payloads = _path_payloads(graph, path)
    return GraphPathResult(
        source_key=node_payloads[0]["key"],
        target_key=node_payloads[-1]["key"],
        path_length=len(path) - 1,
        total_weight=total_weight,
        edges=_edge_steps_from_payloads(edge_payloads),
        steps=_path_steps_from_payloads(node_payloads, edge_payloads),
    )


def _path_payloads(
    graph: rx.PyDiGraph, path: Sequence[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return node and edge payloads for a node-index path."""
    node_payloads = [graph.get_node_data(node_index) for node_index in path]
    edge_payloads = [
        _best_edge_payload(graph, path[i], path[i + 1]) for i in range(len(path) - 1)
    ]
    return node_payloads, edge_payloads


def _path_steps_from_payloads(
    node_payloads: Sequence[dict[str, Any]],
    edge_payloads: Sequence[dict[str, Any]],
) -> list[GraphPathStep]:
    """Create path step models from precomputed payloads."""
    if not node_payloads:
        return []

    steps: list[GraphPathStep] = [
        GraphPathStep(
            kind=cast(GraphPathStepKind, str(node_payloads[0]["kind"])),
            id=node_payloads[0]["id"],
            label=node_payloads[0].get("label"),
        )
    ]

    for step_idx, edge in enumerate(edge_payloads):
        steps.append(_property_step_from_edge(edge))
        target_node = node_payloads[step_idx + 1]
        steps.append(
            GraphPathStep(
                kind=cast(GraphPathStepKind, str(target_node["kind"])),
                id=target_node["id"],
                label=target_node.get("label"),
            )
        )

    return steps


def _edge_steps_from_payloads(
    edge_payloads: Sequence[dict[str, Any]],
) -> list[GraphPathStep]:
    """Return traversed edge steps only."""
    return [_property_step_from_edge(edge) for edge in edge_payloads]


def _property_step_from_edge(edge: dict[str, Any]) -> GraphPathStep:
    """Convert one edge payload into a property path step."""
    return GraphPathStep(
        kind="property",
        id=edge.get("property_id"),
        label=edge.get("property_label"),
        can_be_qualifier=bool(edge.get("property_can_be_qualifier", False)),
        relation=str(edge.get("relation", "unknown")),
        weight=float(edge.get("weight", 1.0)),
    )


def _best_dijkstra_path(
    graph: rx.PyDiGraph,
    source_indices: list[int],
    target_indices: set[int],
    *,
    weighted: bool,
    max_hops: int,
) -> list[int] | None:
    """Return the best source->target path using rustworkx Dijkstra.

    This is used as a fast path for ``k == 1`` while preserving the existing
    formatting and fallback behavior for broader ``k`` enumeration.
    """
    best_path: list[int] | None = None
    best_cost = float("inf")

    def weighted_edge_cost(edge: dict[str, Any]) -> float:
        return float(edge.get("weight", 1.0))

    def unweighted_edge_cost(_edge: dict[str, Any]) -> float:
        return 1.0

    edge_cost_fn = weighted_edge_cost if weighted else unweighted_edge_cost

    for source_index in source_indices:
        try:
            shortest_paths = rx.dijkstra_shortest_paths(
                graph,
                source_index,
                weight_fn=edge_cost_fn,
            )
        except Exception:
            return None

        for target_index in target_indices:
            path = shortest_paths.get(target_index)
            if path is None or len(path) < 2:
                continue
            hops = len(path) - 1
            if hops > max_hops:
                continue
            total_cost = _path_total_cost(graph, path, weighted=weighted)
            if total_cost < best_cost:
                best_cost = total_cost
                best_path = list(path)

    return best_path
