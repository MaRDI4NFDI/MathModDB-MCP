from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from openai import AsyncOpenAI
from rich.console import Console
from typing_extensions import Annotated

from embedder import ColBERTEmbedder, OpenAIEmbedder
from models import MathModDBStructure
from store import QDrantStore

DEFAULT_PREFIXES = {
    "wd": "https://portal.mardi4nfdi.de/entity/",
    "wdt": "https://portal.mardi4nfdi.de/prop/direct/",
}

BASE_DIR = Path.cwd()

load_dotenv(BASE_DIR / ".env")

app = typer.Typer(
    help="MathModDB CLI - Mathematical Modeling Database tools",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main() -> None:
    """MathModDB command group."""


@app.command()
def init(
    graph_dir: Annotated[
        Optional[Path],
        typer.Option(help="Directory to store the graph"),
    ] = None,
    store_dir: Annotated[
        Optional[Path],
        typer.Option(help="Directory to store the store"),
    ] = None,
    enrich: Annotated[
        bool,
        typer.Option(help="Enrich the ontology using GPT"),
    ] = True,
    model: Annotated[
        str,
        typer.Option(help="Model to use for GPT"),
    ] = "gpt-5-mini",
):
    """Initialize the MathModDB ontology and vector store.

    This command downloads the MathModDB ontology from the Wikibase endpoint,
    optionally enriches entity descriptions using GPT, and creates a vector
    store for semantic search.

    Args:
        graph_dir: Directory to cache the ontology graph (default: .graph)
        store_dir: Directory to store the vector database (default: mathmoddb_store)
        enrich: Whether to enrich entity descriptions using GPT
        model: GPT model to use for enrichment (default: gpt-5-mini)
    """

    if graph_dir is None:
        graph_dir = BASE_DIR / ".graph"
    if store_dir is None:
        store_dir = BASE_DIR / "mathmoddb_store"

    console.print("[bold blue]Initializing MathModDB...[/bold blue]")
    console.print(f"Graph directory: {graph_dir}")
    console.print(f"Store directory: {store_dir}")
    console.print(f"Enrichment: {'enabled' if enrich else 'disabled'}")
    if enrich:
        console.print(f"Model: {model}")
    console.print()

    console.print("Loading ontology from Wikibase...")
    ontology = MathModDBStructure.from_wikibase(
        prefixes=DEFAULT_PREFIXES,
        cache_dir=graph_dir,
    )

    console.print("[green]✓[/green] Loaded ontology with:")
    console.print(f"  • {len(ontology.classes)} classes")
    console.print(f"  • {len(ontology.object_properties)} object properties")
    console.print(f"  • {len(ontology.data_properties)} data properties")
    console.print(f"  • {len(ontology.qualifier_properties)} qualifier properties")
    console.print()

    if enrich:
        console.print("Enriching descriptions with GPT...")
        client = AsyncOpenAI()
        ontology.enrich_descriptions(
            client=client,
            model=model,
            max_concurrent_requests=100,
        )
        console.print("[green]✓[/green] Entity descriptions enriched")
        console.print()

    console.print("Initializing vector store...")
    store = QDrantStore(
        db_path=store_dir,
        ontology=ontology,
        dense_embedder=OpenAIEmbedder(),
        multivector_embedder=ColBERTEmbedder(),
    )

    console.print("Embedding ontology entities...")
    store.embed_ontology()

    console.print("[green]✓[/green] Vector store created and populated")
    console.print("[bold green]Initialization complete![/bold green]")
    console.print(f"Vector store saved to: {store_dir}")


if __name__ == "__main__":
    app()
