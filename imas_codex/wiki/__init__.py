"""Wiki ingestion module for facility documentation.

Provides a three-phase pipeline for discovering and ingesting wiki content:

Phase 1 - CRAWL: Fast link extraction, builds wiki graph structure
Phase 2 - SCORE: Agent evaluates graph metrics, assigns interest scores
Phase 3 - INGEST: Fetch content for high-score pages, create chunks

Facility-agnostic design - wiki configuration comes from facility YAML.

Tools are shared between:
- ReAct agents (LlamaIndex FunctionTools)
- MCP server (Cursor chat debugging)

Example:
    from imas_codex.wiki import run_wiki_discovery

    # Run full discovery pipeline
    stats = await run_wiki_discovery(
        facility="epfl",
        cost_limit_usd=10.00,
    )
    print(f"Crawled {stats['pages_crawled']} pages")
"""

from .discovery import (
    DiscoveryStats,
    WikiConfig,
    WikiDiscovery,
    run_wiki_discovery,
)
from .pipeline import (
    WikiIngestionPipeline,
    get_pending_wiki_pages,
    get_wiki_queue_stats,
    mark_wiki_page_status,
)
from .progress import WikiProgressMonitor
from .scraper import (
    WikiPage,
    extract_conventions,
    extract_imas_paths,
    extract_mdsplus_paths,
    extract_units,
    fetch_wiki_page,
)
from .tools import (
    create_wiki_pages,
    fetch_wiki_links,
    fetch_wiki_preview,
    get_graph_schema,
    get_wiki_neighbors,
    get_wiki_pages,
    get_wiki_progress,
    get_wiki_schema,
    get_wiki_tools,
    update_wiki_scores,
)

__all__ = [
    # Discovery pipeline
    "DiscoveryStats",
    "WikiConfig",
    "WikiDiscovery",
    "run_wiki_discovery",
    # Shared tools (ReAct + MCP parity)
    "get_wiki_tools",
    "get_graph_schema",
    "get_wiki_schema",
    "get_wiki_pages",
    "get_wiki_neighbors",
    "get_wiki_progress",
    "update_wiki_scores",
    "create_wiki_pages",
    "fetch_wiki_links",
    "fetch_wiki_preview",
    # Ingestion
    "WikiIngestionPipeline",
    "WikiPage",
    "WikiProgressMonitor",
    # Extraction utilities
    "extract_conventions",
    "extract_imas_paths",
    "extract_mdsplus_paths",
    "extract_units",
    "fetch_wiki_page",
    # Queue management
    "get_pending_wiki_pages",
    "get_wiki_queue_stats",
    "mark_wiki_page_status",
]
