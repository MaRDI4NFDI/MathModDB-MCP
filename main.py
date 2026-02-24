import argparse
from pathlib import Path
from typing import Annotated, Dict

import nest_asyncio
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.utilities.types import Image
from mcp.types import Icon
from toon_format import encode

from src.embedder import ColBERTEmbedder, OpenAIEmbedder
from src.models import MathModDBStructure
from src.sparql import KnowledgeGraph
from src.store import QDrantStore
from src.wikibase import MATHMODDB_WIKIBASE_ENDPOINT

nest_asyncio.apply()


EXPLORE_ONTOLOGY_DESCRIPTION = """STAGE 1 (REQUIRED): discover MathModDB Wikibase schema elements for query planning.

Use this before `sparql_query`. It maps natural language to the IDs and properties you need in SPARQL.

MathModDB is queried as a Wikibase graph:
- Entities/classes use `wd:` IDs (example: `wd:Q6672081`)
- Direct properties use `wdt:` IDs (example: `wdt:P31`)
- Qualifier properties use `pq:` IDs

What this returns (TOON text):
1. Ranked schema candidates:
   - `classes`
   - `object_properties`
   - `data_properties`
   - `qualifier_properties`
   Each item includes core entity metadata and score.
2. A Steiner-style schema snippet for composition:
   - `subgraph`: triples (`subject_class`, `predicate_property`, `object_class`)
   - `data_properties`: dictionary keyed by data property -> list of mapped classes
   - `qualifiers`: dictionary keyed by qualifier -> list of
     `{ "subject_class", "qualified_property" }`

Use the returned IDs/properties directly in SPARQL. This tool returns schema guidance, not full instance retrieval."""

EXPLORE_ONTOLOGY_QUERY_ANNOTATION = """Natural-language schema intent (2-8 words) for MathModDB Wikibase concepts.

Good: "Riemann solvers hyperbolic", "finite volume methods", "enzyme kinetics"
Bad: "math" (too vague), "paper from 2015" (instance-level request; use sparql_query)"""

SPARQL_QUERY_DESCRIPTION = """STAGE 2: query MathModDB Wikibase data with SPARQL.

Prerequisite: run `explore_ontology` first to get valid `wd:`, `wdt:`, and `pq:` IDs.

Rules:
1. Prefixes are preconfigured; do not declare PREFIX blocks.
2. Always include `LIMIT` (recommended <= 50, hard max 100).
3. Prefer small, focused queries and iterate.

Common patterns:
- Instance/class: `?item wdt:P31 wd:QClassID`
- Property filter: `?item wdt:PX ?value`
- English label: `?item rdfs:label ?label . FILTER(LANG(?label) = "en")`

Batch mode is preferred for multiple related requests:
- Pass a dictionary: `{ "name1": "SELECT ...", "name2": "SELECT ..." }`
- Response keys match your dictionary keys.
"""

SPARQL_QUERY_ANNOTATION = """Single SPARQL query string.

SINGLE: "SELECT ?item WHERE { ?item wdt:P31 wd:Q6672081 } LIMIT 10"

Important: Try to use FILTER statements sparingly to avoid rate limiting.

Do not only rely on SPARQL schema discovery, use "explore_ontology" - It is faster!

REQUIREMENTS:
- Include LIMIT (max 100)
- Use codes from explore_ontology (e.g., wdt:P31, wd:Q12345)
- NO PREFIX definitions (automatic)"""

BATCH_SPARQL_QUERY_DESCRIPTION = """Execute multiple SPARQL queries in one call.

Prerequisite: run `explore_ontology` first to get valid `wd:`, `wdt:`, and `pq:` IDs.

Important: Try to use FILTER statements sparingly to avoid rate limiting.

Do not only rely on SPARQL schema discovery, use "explore_ontology" - It is faster!

Rules:
1. Prefixes are preconfigured; do not declare PREFIX blocks.
2. Always include `LIMIT` (recommended <= 50, hard max 100) per query.
3. Query dictionary keys become result keys in the response.
"""

# Default prefixes for the MathModDB ontology
DEFAULT_PREFIXES = {
    "wd": "https://portal.mardi4nfdi.de/entity/",
    "wdt": "https://portal.mardi4nfdi.de/prop/direct/",
}

# Common paths
BASE_DIR = Path(__file__).parent

# Load OPENAI_API_KEY from .env file
load_dotenv(BASE_DIR / ".env")

# Load ontology and knowledge graph
ontology = MathModDBStructure.from_wikibase(
    prefixes=DEFAULT_PREFIXES,
    cache_dir=BASE_DIR / ".graph",
)
kg = KnowledgeGraph(endpoint=MATHMODDB_WIKIBASE_ENDPOINT)

# Initialize embedders
embedder = OpenAIEmbedder()
multivector_embedder = ColBERTEmbedder()

# Initialize vector database store
store = QDrantStore(
    db_path=BASE_DIR / "mathmoddb_store",
    ontology=ontology,
    dense_embedder=embedder,
    multivector_embedder=multivector_embedder,
)

# Initialize FastMCP app
icon = Image(path=BASE_DIR / "assets/icon.png")
app = FastMCP(
    "MathModDB",
    version="0.1.0",
    website_url="https://github.com/MaRDI4NFDI/MathModDB-MCP",
    icons=[
        Icon(src=icon.to_data_uri(), mimeType="image/png", sizes=["96x96"]),
    ],
)


@app.tool(
    name="Explore_Ontology",
    description=EXPLORE_ONTOLOGY_DESCRIPTION,
    tags={"ontology", "discovery", "exploration", "search", "concepts"},
)
def explore_ontology(
    query: Annotated[
        str,
        EXPLORE_ONTOLOGY_QUERY_ANNOTATION,
    ],
) -> str:
    """
    Explore MathModDB Wikibase schema using semantic search.

    Args:
        query: Natural-language concept intent for schema discovery.

    Returns:
        TOON-encoded string containing:
        1) Ranked candidates (`classes`, `object_properties`, `data_properties`,
           `qualifier_properties`) with IDs/scores.
        2) A connected schema snippet with:
           - `subgraph` triples,
           - `data_properties` as data_property -> list[class],
           - `qualifiers` as qualifier -> list[{subject_class, qualified_property}].
    """

    return store.search_all(query, k=6).toon()


@app.tool(
    name="SPARQL_Query",
    description=SPARQL_QUERY_DESCRIPTION,
    tags={"sparql", "query", "data", "retrieval", "knowledge-graph"},
)
def sparql_query(
    query: Annotated[
        str,
        SPARQL_QUERY_ANNOTATION,
    ],
) -> str:
    """
    Execute SPARQL queries against the MathModDB knowledge graph.

    This tool provides direct access to the RDF data using SPARQL queries.
    It executes one query per call and supports all SPARQL query types
    (SELECT, ASK, CONSTRUCT, DESCRIBE).

    Best practices:
    - Always include a LIMIT clause (maximum 100 results per query)
    - Use explore_ontology first to understand available concepts
    - Start with simple SELECT queries to understand the data structure

    Common query patterns:
    - Find all classes: SELECT ?class WHERE { ?class a owl:Class } LIMIT 10
    - Find instances: SELECT ?instance WHERE { ?instance a ?class } LIMIT 10
    - Find properties: SELECT ?prop WHERE { ?prop a owl:ObjectProperty } LIMIT 10
    - Find relationships: SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10

    Args:
        query: Single SPARQL query string

    Returns:
        Query results formatted as structured data. For SELECT queries, returns
        variable bindings. For other query types, returns appropriate result format.
        Errors are captured and returned in the response.
    """
    return encode(kg.query(query))


@app.tool(
    name="Batch_SPARQL_Query",
    description=BATCH_SPARQL_QUERY_DESCRIPTION,
    tags={"sparql", "query", "batch", "data", "retrieval", "knowledge-graph"},
)
def batch_sparql_query(
    queries: Annotated[
        Dict[str, str],
        "Dictionary of named SPARQL queries. Keys become result names. Example: {'classes': 'SELECT ?class WHERE { ?class a owl:Class } LIMIT 10', 'properties': 'SELECT ?prop WHERE { ?prop a owl:ObjectProperty } LIMIT 10'}",
    ],
) -> str:
    """
    Execute multiple named SPARQL queries against the MathModDB knowledge graph.

    Use this tool when several related queries are needed in one request.
    Each dictionary key is preserved in the response as the corresponding
    result key.

    Args:
        queries: Dictionary of query-name -> SPARQL query string

    Returns:
        Batch query results encoded as TOON text.
    """
    return encode(kg.query(queries))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--port", type=int, default=8000)
    args.add_argument("--host", type=str, default="0.0.0.0")
    args.add_argument("--dry-run", action="store_true")
    args = args.parse_args()

    if args.dry_run:
        print("Dry run complete")
        exit(0)

    host = args.host
    port = args.port

    """
    Entry point for the MathModDB MCP server.
    
    Starts the FastMCP server which exposes the ontology exploration and SPARQL
    query tools via the Model Context Protocol. The server runs with the pre-loaded
    MathModDB ontology and initialized vector database.
    
    The server will be available for MCP clients to connect and use the exposed tools
    for exploring mathematical modeling concepts and querying the knowledge graph.
    """
    app.run(transport="streamable-http", host=host, port=port)
