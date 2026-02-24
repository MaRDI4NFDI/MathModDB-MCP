from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import List

import httpx
import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from models import (
    Class,
    Connection,
    DataProperty,
    MathModDBStructure,
    ObjectProperty,
    QualifierPath,
    QualifierProperty,
)

MATHMODDB_MARKER_PROP = "wdt:P1495"
MATHMODDB_MARKER_ITEM = "wd:Q6534265"
MATHMODDB_WIKIBASE_ENDPOINT = "https://query.portal.mardi4nfdi.de/sparql"
MATHMODDB_PROP_NAMESPACE = "https://portal.mardi4nfdi.de/prop/"
MATHMODDB_QUALIFIER_NAMESPACE = f"{MATHMODDB_PROP_NAMESPACE}qualifier/"
MATHMODDB_ENTITY_NAMESPACE = "https://portal.mardi4nfdi.de/entity/"
QUALIFIER_PREDICATE_PLACEHOLDER = "__QUALIFIER_PREDICATE_IRI__"

# Some qualifier entities can also appear in object/data-property query results.
# We filter those from object properties when the description explicitly marks
# qualifier usage.
QUALIFIER_DESCRIPTION_FILTER = "(qualifier)"

# Logger used by retry hooks.
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

console = Console()

# Cache directory under the system temporary directory.
CACHE_DIR = Path(tempfile.gettempdir()) / "mardi-mcp" / "wikibase_cache"


# --------------------------------------------------
# SPARQL QUERIES
# --------------------------------------------------

SPARQL_CLASSES = f"""
SELECT ?id ?label ?description (COUNT(?x) AS ?usageCount)
WHERE {{
  ?x {MATHMODDB_MARKER_PROP} {MATHMODDB_MARKER_ITEM} ;
     wdt:P31 ?id .

  ?id schema:description ?description .
  FILTER(lang(?description) = "en")

  OPTIONAL {{ ?id rdfs:label ?label . FILTER(lang(?label) = "en") }}
}}
GROUP BY ?id ?label ?description
"""

SPARQL_CLASS_PROPERTIES_TEMPLATE = """
SELECT
  ?propertyEntity
  ?propLabel
  ?propDescription
  ?propertyType
  (COUNT(?value) AS ?usageCount)
WHERE {{
  ?expression wdt:P31 <{class_iri}> ;
              ?propertyEntity ?value .

  # Map direct predicate → property entity (only for metadata)
  ?propEntity wikibase:directClaim ?propertyEntity ;
              wikibase:propertyType ?propertyType .

  OPTIONAL {{
    ?propEntity rdfs:label ?propLabel .
    FILTER(lang(?propLabel) = "en")
  }}
  OPTIONAL {{
    ?propEntity schema:description ?propDescription .
    FILTER(lang(?propDescription) = "en")
  }}

  # Exclude external identifiers
  FILTER(?propertyType != wikibase:ExternalId)
}}
GROUP BY ?propertyEntity ?propLabel ?propDescription ?propertyType
"""

SPARQL_OBJ_PROPERTIES_RESOLUTION = """
SELECT
  ?sourceClass
  ?sourceClassLabel
  ?targetClass
  ?targetClassLabel
  (COUNT(*) AS ?usageCount)
WHERE {{
  ?source {mathmoddb_marker_prop} {mathmoddb_marker_item} ;
          <{object_property_iri}> ?target ;
          wdt:P31 ?sourceClass .

  ?target {mathmoddb_marker_prop} {mathmoddb_marker_item} ;
          wdt:P31 ?targetClass .
}}
GROUP BY ?sourceClass ?sourceClassLabel ?targetClass ?targetClassLabel
"""

SPARQL_QUALIFIER_PROPERTIES = f"""
SELECT ?qualifier ?label ?description (COUNT(*) AS ?usageCount)
WHERE {{
  ?x {MATHMODDB_MARKER_PROP} {MATHMODDB_MARKER_ITEM} .
  ?x ?p ?statement .
  FILTER(STRSTARTS(STR(?p), "{MATHMODDB_PROP_NAMESPACE}"))

  ?statement ?pq ?qualifierValue .
  FILTER(STRSTARTS(STR(?pq), "{MATHMODDB_QUALIFIER_NAMESPACE}"))

  BIND(
    IRI(
      REPLACE(
        STR(?pq),
        "{MATHMODDB_QUALIFIER_NAMESPACE}",
        "{MATHMODDB_ENTITY_NAMESPACE}"
      )
    ) AS ?qualifier
  )

  OPTIONAL {{
    ?qualifier rdfs:label ?label .
    FILTER(lang(?label) = "en")
  }}
  OPTIONAL {{
    ?qualifier schema:description ?description .
    FILTER(lang(?description) = "en")
  }}
}}
GROUP BY ?qualifier ?label ?description
"""

SPARQL_QUALIFIER_PATHS_TEMPLATE = f"""
SELECT ?mainProperty ?subjectClass (COUNT(*) AS ?usageCount)
WHERE {{
  ?x {MATHMODDB_MARKER_PROP} {MATHMODDB_MARKER_ITEM} .
  ?x ?p ?statement .
  FILTER(STRSTARTS(STR(?p), "{MATHMODDB_PROP_NAMESPACE}"))

  ?statement <{QUALIFIER_PREDICATE_PLACEHOLDER}> ?qualifierValue .

  BIND(
    IRI(
      REPLACE(
        STR(?p),
        "{MATHMODDB_PROP_NAMESPACE}",
        "{MATHMODDB_ENTITY_NAMESPACE}"
      )
    ) AS ?mainProperty
  )

  OPTIONAL {{ ?x wdt:P31 ?subjectClass . }}
}}
GROUP BY ?mainProperty ?subjectClass
"""


async def initialize_kg_from_wikibase(
    endpoint: str = MATHMODDB_WIKIBASE_ENDPOINT,
    max_concurrent_requests: int = 2,
    sleep_seconds: float = 1.0,
    refresh: bool = False,
    initial_prefixes: dict[str, str] | None = None,
    cache_dir: str | Path | None = None,
) -> MathModDBStructure:
    """Initialize MathModDBStructure from WikiBase SPARQL endpoint.

    This function fetches classes and their properties from a WikiBase instance
    and constructs a complete knowledge graph structure. Results are cached to
    avoid repeated fetches.

    Args:
        endpoint: SPARQL endpoint URL
        max_concurrent_requests: Maximum number of concurrent SPARQL requests
        sleep_seconds: Sleep duration between requests
        refresh: If True, ignore cache and fetch fresh data
        initial_prefixes: Optional seed prefixes used during IRI resolution.
        cache_dir: Optional custom cache directory. Uses default temp cache when None.

    Returns:
        MathModDBStructure instance populated with classes and properties
    """
    # Check cache first unless refresh is requested
    cache_path = _get_cache_path(endpoint, cache_dir=cache_dir)
    if not refresh:
        cached_kg = _load_from_cache(cache_path)
        if cached_kg is not None:
            return cached_kg

    # Initialize property dictionaries (shared across all classes)
    data_props_dict: dict[str, DataProperty] = {}
    obj_props_dict: dict[str, ObjectProperty] = {}

    # Cache for object property connections (keyed by property IRI)
    obj_prop_connections_cache: dict[str, list[Connection]] = {}

    # Create locks for thread-safe dictionary access
    data_lock = asyncio.Lock()
    obj_lock = asyncio.Lock()
    connections_cache_lock = asyncio.Lock()

    async with httpx.AsyncClient() as client:
        # Fetch classes first
        classes_raw = await fetch_classes(client, endpoint, sleep_seconds)

        # Process classes with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Processing classes...", total=len(classes_raw))

            # Create semaphore to limit concurrent requests for class processing
            semaphore = asyncio.Semaphore(max_concurrent_requests)

            # Create separate semaphore for object property connection fetching
            # Use higher concurrency for connection fetching (5x the class processing limit)
            connections_semaphore = asyncio.Semaphore(max_concurrent_requests * 5)

            # Create temporary structure for prefix resolution during processing
            # We'll create the final structure after all classes are processed
            temp_structure = MathModDBStructure(
                prefixes=initial_prefixes.copy() if initial_prefixes else {},
            )

            async def process_with_semaphore(cls: dict) -> Class:
                async with semaphore:
                    return await process_class(
                        client,
                        cls,
                        endpoint,
                        sleep_seconds,
                        progress,
                        task_id,
                        data_props_dict,
                        obj_props_dict,
                        data_lock,
                        obj_lock,
                        temp_structure,
                        obj_prop_connections_cache,
                        connections_semaphore,
                        connections_cache_lock,
                    )

            # Process all classes concurrently
            classes = await asyncio.gather(
                *[process_with_semaphore(cls) for cls in classes_raw]
            )

            qualifier_properties = await fetch_qualifier_properties(
                client=client,
                endpoint=endpoint,
                sleep_seconds=sleep_seconds,
                structure=temp_structure,
                progress=progress,
            )

    # We need to filter out qualifier properties from object properties.
    obj_props = []

    for prop in obj_props_dict.values():
        if prop.description is None:
            obj_props.append(prop)
        elif QUALIFIER_DESCRIPTION_FILTER not in prop.description.lower():
            obj_props.append(prop)
        else:
            continue

    # Create knowledge graph with separate lists per entity type
    # Use prefixes from temp_structure which were populated during processing
    kg = MathModDBStructure(
        prefixes=temp_structure.prefixes.copy(),
        classes=classes,
        object_properties=obj_props,
        data_properties=list(data_props_dict.values()),
        qualifier_properties=qualifier_properties,
    )

    # Save to cache
    _save_to_cache(kg, cache_path)

    return kg


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
    before_sleep=before_sleep_log(logger, logging.INFO),
)
async def run_sparql(
    client: httpx.AsyncClient,
    query: str,
    endpoint: str,
    sleep_seconds: float,
) -> dict:
    """Execute a SPARQL query with retry logic.

    Args:
        client: HTTP client for making requests
        query: SPARQL query string
        endpoint: SPARQL endpoint URL
        sleep_seconds: Sleep duration before request

    Returns:
        JSON response from SPARQL endpoint

    Raises:
        RuntimeError: If SPARQL query returns 400 error
        httpx.HTTPStatusError: If request fails after retries
    """
    await asyncio.sleep(sleep_seconds)
    headers = {"Accept": "application/sparql-results+json"}
    response = await client.get(
        endpoint,
        params={"query": query, "format": "json"},
        headers=headers,
        timeout=60,
    )
    if response.status_code == 400:
        raise RuntimeError(f"SPARQL error:\n{query}")
    response.raise_for_status()
    return response.json()


async def fetch_classes(
    client: httpx.AsyncClient,
    endpoint: str,
    sleep_seconds: float,
) -> List[dict]:
    """Fetch all classes from WikiBase that match MathModDB marker.

    Args:
        client: HTTP client for making requests
        endpoint: SPARQL endpoint URL
        sleep_seconds: Sleep duration between requests

    Returns:
        List of class dictionaries with iri, id, label, and comment fields
    """
    res = await run_sparql(client, SPARQL_CLASSES, endpoint, sleep_seconds)
    classes = [
        {
            "iri": b["id"]["value"],
            "id": iri_to_id(b["id"]["value"]),
            "label": b.get("label", {}).get("value"),
            "comment": b["description"]["value"],
            "quantity": int(b["usageCount"]["value"]),
        }
        for b in res["results"]["bindings"]
    ]
    return classes


async def fetch_qualifier_properties(
    client: httpx.AsyncClient,
    endpoint: str,
    sleep_seconds: float,
    structure: MathModDBStructure,
    progress: Progress | None = None,
) -> list[QualifierProperty]:
    """Fetch qualifier properties and their subject/property usage paths."""
    res = await run_sparql(client, SPARQL_QUALIFIER_PROPERTIES, endpoint, sleep_seconds)

    qualifier_props: dict[str, QualifierProperty] = {}

    for b in res["results"]["bindings"]:
        qualifier_iri = b["qualifier"]["value"]
        qualifier_id = iri_to_id(qualifier_iri)
        qualifier_prefix = structure.resolve_prefix(qualifier_iri)

        if qualifier_id in qualifier_props:
            continue

        qualifier_props[qualifier_id] = QualifierProperty(
            prefix=qualifier_prefix,
            id=qualifier_id,
            label=b.get("label", {}).get("value"),
            description=b.get("description", {}).get("value"),
            quantity=int(b["usageCount"]["value"]),
            structure=structure,
        )

    qualifier_task_id: TaskID | None = None
    if progress is not None:
        qualifier_task_id = progress.add_task(
            "Resolving qualifier paths...",
            total=len(qualifier_props),
        )

    for qualifier in qualifier_props.values():
        qualifier_predicate_iri = (
            f"{MATHMODDB_QUALIFIER_NAMESPACE}{qualifier.id}" if qualifier.id else None
        )

        if qualifier_predicate_iri is None:
            if progress is not None and qualifier_task_id is not None:
                progress.update(qualifier_task_id, advance=1)
            continue

        query = SPARQL_QUALIFIER_PATHS_TEMPLATE.replace(
            QUALIFIER_PREDICATE_PLACEHOLDER,
            qualifier_predicate_iri,
        )
        path_res = await run_sparql(client, query, endpoint, sleep_seconds)

        for b in path_res["results"]["bindings"]:
            subject_class_iri = b.get("subjectClass", {}).get("value")
            main_property_iri = b["mainProperty"]["value"]

            if not subject_class_iri:
                continue

            subject_class_id = iri_to_id(subject_class_iri)
            property_id = iri_to_id(main_property_iri)
            usage_count = int(b["usageCount"]["value"])

            qualifier.qualifier_paths.append(
                QualifierPath(
                    subject_class_id=subject_class_id,
                    property_id=property_id,
                    usage_count=usage_count,
                )
            )

        if progress is not None and qualifier_task_id is not None:
            progress.update(qualifier_task_id, advance=1)

    return list(qualifier_props.values())


async def fetch_class_properties(
    client: httpx.AsyncClient,
    class_iri: str,
    class_id: str,
    endpoint: str,
    sleep_seconds: float,
    data_props_dict: dict[str, DataProperty],
    obj_props_dict: dict[str, ObjectProperty],
    data_lock: asyncio.Lock,
    obj_lock: asyncio.Lock,
    structure: MathModDBStructure,
    obj_prop_connections_cache: dict[str, list[Connection]],
    connections_semaphore: asyncio.Semaphore,
    connections_cache_lock: asyncio.Lock,
) -> tuple[List[str], List[str]]:
    """Fetch properties for a class and update property dictionaries.

    Args:
        client: HTTP client for making requests
        class_iri: Full IRI of the class
        class_id: ID of the class
        endpoint: SPARQL endpoint URL
        sleep_seconds: Sleep duration between requests
        data_props_dict: Dictionary of data properties (mutated)
        obj_props_dict: Dictionary of object properties (mutated)
        data_lock: Lock for thread-safe access to data_props_dict
        obj_lock: Lock for thread-safe access to obj_props_dict
        structure: Ontology structure used for prefix resolution.
        obj_prop_connections_cache: Cached object-property connection lists.
        connections_semaphore: Concurrency limit for connection fetches.
        connections_cache_lock: Lock guarding the connection cache.

    Returns:
        Tuple of (data_property_ids, object_property_ids) for this class
    """
    query = SPARQL_CLASS_PROPERTIES_TEMPLATE.format(class_iri=class_iri)
    res = await run_sparql(client, query, endpoint, sleep_seconds)

    data_prop_ids: list[str] = []
    obj_prop_ids: list[str] = []

    # Collect object properties that need connection fetching
    obj_props_to_fetch: list[
        tuple[str, str, str, str]
    ] = []  # (prop_iri, prop_id, prop_label, prop_comment)

    for b in res["results"]["bindings"]:
        prop_iri = b["propertyEntity"]["value"]
        prop_id = iri_to_id(prop_iri)

        prop_type = b["propertyType"]["value"]
        is_object_prop = prop_type == "http://wikiba.se/ontology#WikibaseItem"
        usage_count = int(b["usageCount"]["value"])

        # Get the appropriate dictionary and lock
        lock = obj_lock if is_object_prop else data_lock

        # Create or update property (with lock for thread safety)
        async with lock:
            prop_label = b.get("propLabel", {}).get("value")
            prop_comment = b.get("propDescription", {}).get("value")

            # Resolve prefix from IRI
            prop_prefix = structure.resolve_prefix(prop_iri)

            if is_object_prop:
                if prop_id not in obj_props_dict:
                    # Collect for batch fetching
                    obj_props_to_fetch.append(
                        (prop_iri, prop_id, prop_label, prop_comment)
                    )
                else:
                    existing_quantity = obj_props_dict[prop_id].quantity or 0
                    obj_props_dict[prop_id].quantity = existing_quantity + usage_count
                    # Add class_id if not already present
                    if class_id not in obj_props_dict[prop_id].class_ids:
                        obj_props_dict[prop_id].class_ids.append(class_id)
            else:
                if prop_id not in data_props_dict:
                    data_props_dict[prop_id] = DataProperty(
                        prefix=prop_prefix,
                        id=prop_id,
                        label=prop_label,
                        description=prop_comment,
                        quantity=usage_count,
                        class_ids=[class_id],
                        structure=structure,
                    )
                else:
                    existing_quantity = data_props_dict[prop_id].quantity or 0
                    data_props_dict[prop_id].quantity = existing_quantity + usage_count
                    # Add class_id if not already present
                    if class_id not in data_props_dict[prop_id].class_ids:
                        data_props_dict[prop_id].class_ids.append(class_id)

        # Track property ID for this class
        if is_object_prop:
            if prop_id not in obj_prop_ids:
                obj_prop_ids.append(prop_id)
        else:
            if prop_id not in data_prop_ids:
                data_prop_ids.append(prop_id)

    # Batch fetch object property connections concurrently
    if obj_props_to_fetch:
        # Filter out properties that are already cached
        props_to_fetch: list[tuple[str, str, str, str]] = []
        cached_connections: dict[str, list[Connection]] = {}

        async with connections_cache_lock:
            for prop_iri, prop_id, prop_label, prop_comment in obj_props_to_fetch:
                if prop_iri in obj_prop_connections_cache:
                    cached_connections[prop_iri] = obj_prop_connections_cache[prop_iri]
                else:
                    props_to_fetch.append((prop_iri, prop_id, prop_label, prop_comment))

        # Fetch missing connections concurrently
        if props_to_fetch:

            async def fetch_with_semaphore(
                prop_iri: str,
            ) -> tuple[str, list[Connection]]:
                async with connections_semaphore:
                    # Double-check cache after acquiring semaphore (another task might have fetched it)
                    async with connections_cache_lock:
                        if prop_iri in obj_prop_connections_cache:
                            return (prop_iri, obj_prop_connections_cache[prop_iri])

                    # Fetch connections
                    connections = await _process_object_properties(
                        client,
                        prop_iri,
                        endpoint,
                        sleep_seconds,
                    )

                    # Update cache
                    async with connections_cache_lock:
                        obj_prop_connections_cache[prop_iri] = connections

                    return (prop_iri, connections)

            # Fetch all missing connections concurrently
            fetched_results = await asyncio.gather(
                *[
                    fetch_with_semaphore(prop_iri)
                    for prop_iri, _, _, _ in props_to_fetch
                ]
            )

            # Merge fetched results into cache dict
            for prop_iri, connections in fetched_results:
                cached_connections[prop_iri] = connections

        # Create ObjectProperty instances with cached/fetched connections
        async with obj_lock:
            for prop_iri, prop_id, prop_label, prop_comment in obj_props_to_fetch:
                if prop_id not in obj_props_dict:
                    prop_prefix = structure.resolve_prefix(prop_iri)
                    connections = cached_connections[prop_iri]
                    obj_props_dict[prop_id] = ObjectProperty(
                        prefix=prop_prefix,
                        id=prop_id,
                        label=prop_label,
                        description=prop_comment,
                        quantity=usage_count,
                        class_ids=[class_id],
                        structure=structure,
                        common_connections=connections,
                    )
                else:
                    existing_quantity = obj_props_dict[prop_id].quantity or 0
                    obj_props_dict[prop_id].quantity = existing_quantity + usage_count
                    if class_id not in obj_props_dict[prop_id].class_ids:
                        obj_props_dict[prop_id].class_ids.append(class_id)

    return data_prop_ids, obj_prop_ids


async def process_class(
    client: httpx.AsyncClient,
    cls: dict,
    endpoint: str,
    sleep_seconds: float,
    progress: Progress,
    task_id: TaskID,
    data_props_dict: dict[str, DataProperty],
    obj_props_dict: dict[str, ObjectProperty],
    data_lock: asyncio.Lock,
    obj_lock: asyncio.Lock,
    structure: MathModDBStructure,
    obj_prop_connections_cache: dict[str, list[Connection]],
    connections_semaphore: asyncio.Semaphore,
    connections_cache_lock: asyncio.Lock,
) -> Class:
    """Process a single class and update progress.

    Args:
        client: HTTP client for making requests
        cls: Class dictionary with iri, id, label, comment
        endpoint: SPARQL endpoint URL
        sleep_seconds: Sleep duration between requests
        progress: Progress bar instance
        task_id: Task ID for progress tracking
        data_props_dict: Dictionary of data properties (mutated)
        obj_props_dict: Dictionary of object properties (mutated)
        data_lock: Lock for thread-safe access to data_props_dict
        obj_lock: Lock for thread-safe access to obj_props_dict
        structure: Ontology structure used for prefix resolution.
        obj_prop_connections_cache: Cached object-property connection lists.
        connections_semaphore: Concurrency limit for connection fetches.
        connections_cache_lock: Lock guarding the connection cache.

    Returns:
        Class instance with properties populated
    """
    class_id = cls["id"]
    progress.update(task_id, description=f"Processing {class_id}")

    data_prop_ids, obj_prop_ids = await fetch_class_properties(
        client,
        cls["iri"],
        class_id,
        endpoint,
        sleep_seconds,
        data_props_dict,
        obj_props_dict,
        data_lock,
        obj_lock,
        structure,
        obj_prop_connections_cache,
        connections_semaphore,
        connections_cache_lock,
    )

    progress.advance(task_id)

    # Resolve prefix from IRI
    class_prefix = structure.resolve_prefix(cls["iri"])

    return Class(
        prefix=class_prefix,
        id=class_id,
        label=cls["label"],
        description=cls["comment"],
        quantity=cls["quantity"],
        object_property_ids=obj_prop_ids,
        data_property_ids=data_prop_ids,
        structure=structure,
    )


async def _process_object_properties(
    client: httpx.AsyncClient,
    object_property_iri: str,
    endpoint: str,
    sleep_seconds: float,
) -> list[Connection]:
    """Resolve class-to-class connections for one object property IRI.

    Returns:
        List of `Connection` objects with identifier-only class IDs.
    """
    query = SPARQL_OBJ_PROPERTIES_RESOLUTION.format(
        object_property_iri=object_property_iri,
        mathmoddb_marker_prop=MATHMODDB_MARKER_PROP,
        mathmoddb_marker_item=MATHMODDB_MARKER_ITEM,
    )
    res = await run_sparql(client, query, endpoint, sleep_seconds)

    return [
        Connection(
            subject_id=iri_to_id(b["sourceClass"]["value"]),
            object_id=iri_to_id(b["targetClass"]["value"]),
            usage_count=int(b["usageCount"]["value"]),
        )
        for b in res["results"]["bindings"]
    ]


def _get_cache_path(endpoint: str, cache_dir: str | Path | None = None) -> Path:
    """Generate cache file path based on endpoint URL.

    Args:
        endpoint: SPARQL endpoint URL
        cache_dir: Optional custom cache directory

    Returns:
        Absolute Path to cache file
    """
    # Build a deterministic cache filename from the endpoint URL.
    import hashlib

    endpoint_hash = hashlib.md5(endpoint.encode()).hexdigest()[:8]
    base_cache_dir = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    cache_path = base_cache_dir / f"wikibase_kg_{endpoint_hash}.yaml"
    return cache_path.resolve()


def _load_from_cache(cache_path: Path) -> MathModDBStructure | None:
    """Load MathModDBStructure from cache file if it exists.

    Args:
        cache_path: Path to cache file

    Returns:
        MathModDBStructure instance if cache exists, None otherwise
    """
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        kg = MathModDBStructure.model_validate(data)
        return kg
    except Exception:
        return None


def _save_to_cache(kg: MathModDBStructure, cache_path: Path) -> None:
    """Save MathModDBStructure to cache file.

    Args:
        kg: MathModDBStructure instance to save
        cache_path: Path to cache file
    """
    # Create cache directory if it doesn't exist
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_path, "w", encoding="utf-8") as f:
        yaml.dump(
            kg.model_dump(),
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )


def iri_to_id(iri: str) -> str:
    """Extract ID from IRI by taking the last segment after the final slash.

    Args:
        iri: Full IRI string

    Returns:
        ID string (e.g., "Q1234567" or "P123")
    """
    return iri.rsplit("/", 1)[-1]
