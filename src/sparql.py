from typing import Any, Dict, List, Optional, Union

from SPARQLWrapper import JSON, SPARQLWrapper

# Type alias for query results
QueryResult = Union[List[Dict[str, Any]], Dict[str, Any]]


class KnowledgeGraph:
    """
    A high-level interface for querying RDF knowledge graphs using SPARQL.

    This class provides convenient methods for executing SPARQL queries against
    a SPARQL endpoint with automatic error handling and result formatting.

    Attributes:
        endpoint (str): The SPARQL endpoint URL to query.

    Example:
        >>> kg = KnowledgeGraph(endpoint="https://query.portal.mardi4nfdi.de/sparql")
        >>> results = kg.query("SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10")
    """

    def __init__(self, endpoint: str):
        """
        Initialize the KnowledgeGraph with a SPARQL endpoint URL.

        Args:
            endpoint (str): The SPARQL endpoint URL to query against.
        """
        self.endpoint = endpoint

    def query(
        self,
        query: Union[str, Dict[str, str]],
    ) -> Union[QueryResult, Dict[str, QueryResult]]:
        """
        Execute one or more SPARQL queries against the knowledge graph.

        This method can handle both single queries (passed as strings) and
        batch queries (passed as a dictionary mapping names to query strings).

        Args:
            query: Either a single SPARQL query string or a dictionary mapping
                  query names to SPARQL query strings for batch execution.

        Returns:
            For single queries: A QueryResult (list of dicts or error dict).
            For batch queries: A dictionary mapping query names to their results.

        Example:
            Single query:
            >>> kg.query("SELECT ?s WHERE { ?s a owl:Class }")

            Batch queries:
            >>> queries = {
            ...     "classes": "SELECT ?s WHERE { ?s a owl:Class }",
            ...     "properties": "SELECT ?s WHERE { ?s a owl:ObjectProperty }"
            ... }
            >>> kg.query(queries)
        """
        if isinstance(query, str):
            return _execute_single_query(self.endpoint, query)
        return _execute_batch_queries(self.endpoint, query)


def _execute_single_query(endpoint: str, query: str) -> QueryResult:
    """
    Execute a single SPARQL query and return results.

    Args:
        endpoint: The SPARQL endpoint URL to query.
        query: The SPARQL query string to execute.

    Returns:
        Either a list of result dictionaries (on success) or an error dictionary
        with an "error" key (on failure).
    """
    try:
        wrapper = SPARQLWrapper(endpoint)
        wrapper.setQuery(query)
        wrapper.setReturnFormat(JSON)
        results = wrapper.query().convert()
        return _sparql_results_to_dicts(results)
    except Exception as e:
        return {"error": str(e)}


def _execute_batch_queries(
    endpoint: str, queries: Dict[str, str]
) -> Dict[str, QueryResult]:
    """
    Execute multiple SPARQL queries and return named results.

    Each query is executed independently, and errors are captured per query
    without stopping execution of others.

    Args:
        endpoint: The SPARQL endpoint URL to query.
        queries: A dictionary mapping query names to SPARQL query strings.

    Returns:
        A dictionary mapping query names to their results. Each result is either
        a list of result dictionaries (on success) or an error dictionary with
        an "error" key (on failure).
    """
    return {
        name: _execute_single_query(endpoint, query) for name, query in queries.items()
    }


def _sparql_results_to_dicts(results: Any) -> List[Dict[str, Any]]:
    """
    Convert SPARQL query results (JSON format) to a list of dictionaries.

    Args:
        results: The SPARQL query result dictionary from SPARQLWrapper.

    Returns:
        A list of dictionaries representing the query results. For SELECT
        queries, each dictionary maps variable names to values. For ASK
        queries, returns a single-item list with a `{"boolean": ...}` dict.
    """
    # Handle ASK queries (boolean results)
    if "boolean" in results:
        return [{"boolean": results["boolean"]}]

    # Handle SELECT queries
    if "results" in results and "bindings" in results["results"]:
        bindings = results["results"]["bindings"]
        if not bindings:
            return []

        # Extract variable names from the first binding
        if bindings:
            return [
                {
                    var: _format_value(binding[var]["value"])
                    for var in binding.keys()
                    if var in binding
                }
                for binding in bindings
            ]

    return []


def _format_value(value: Any) -> Optional[str]:
    """
    Format a SPARQL result value to string, handling None.

    Args:
        value: The value to format. Can be any type, including None.

    Returns:
        The string representation of the value, or None if the input was None.
    """
    return str(value) if value is not None else None
