"""Wiki discovery tools shared between ReAct agents and MCP.

These tools provide parity between:
1. LlamaIndex ReActAgent tools (for autonomous discovery)
2. MCP tools (for Cursor chat debugging and manual control)

The same prompts and tools work in both contexts, enabling:
- Automated discovery via `imas-codex wiki discover`
- Interactive debugging via Cursor chat with MCP tools
"""

import json
import logging
import subprocess
import urllib.parse

from llama_index.core.tools import FunctionTool

from imas_codex.graph import GraphClient, get_schema

logger = logging.getLogger(__name__)

# =============================================================================
# Schema Tool - Returns focused subset for wiki operations
# =============================================================================


def get_wiki_schema(focus: str = "all") -> str:
    """Get WikiPage schema for graph writes.

    Args:
        focus: Schema subset to return:
            - "create": Fields needed for creating WikiPage nodes
            - "score": Fields for scoring/evaluating pages
            - "status": Valid status values and transitions
            - "all": Complete WikiPage schema

    Returns:
        JSON schema with field definitions, types, and valid values.
    """
    # Status enum values
    status_values = [
        "crawled",  # Link extracted, awaiting scoring
        "scored",  # Interest score assigned
        "discovered",  # High-score, ready for ingestion
        "skipped",  # Low-score with skip_reason
        "ingested",  # Content fetched and chunked
        "failed",  # Error during processing
        "stale",  # May need re-crawling
    ]

    # Field definitions
    all_fields = {
        "id": {
            "type": "string",
            "required": True,
            "description": "Composite key: facility:page_name (e.g., 'epfl:Thomson')",
            "example": "epfl:Thomson",
        },
        "facility_id": {
            "type": "string",
            "required": True,
            "description": "Parent facility ID",
            "example": "epfl",
        },
        "url": {
            "type": "string",
            "required": True,
            "description": "Full wiki URL",
        },
        "title": {
            "type": "string",
            "required": True,
            "description": "Page title (may differ from page_name)",
        },
        "status": {
            "type": "enum",
            "required": True,
            "values": status_values,
            "description": "Lifecycle status",
        },
        "link_depth": {
            "type": "integer",
            "description": "Distance from portal (0=portal, 1=direct link)",
        },
        "in_degree": {
            "type": "integer",
            "description": "Number of pages linking TO this page",
        },
        "out_degree": {
            "type": "integer",
            "description": "Number of pages this links TO",
        },
        "interest_score": {
            "type": "float",
            "range": [0.0, 1.0],
            "description": "Agent-assigned priority (higher=more interesting)",
        },
        "score_reasoning": {
            "type": "string",
            "description": "Agent's explanation for the score",
        },
        "skip_reason": {
            "type": "string",
            "description": "Why page was skipped (if status=skipped)",
            "examples": ["in_degree=0, orphan page", "administrative content"],
        },
        "discovered_at": {
            "type": "datetime",
            "description": "When first crawled",
        },
        "scored_at": {
            "type": "datetime",
            "description": "When interest_score assigned",
        },
    }

    # Return focused subset
    if focus == "create":
        fields = {
            k: all_fields[k]
            for k in [
                "id",
                "facility_id",
                "url",
                "title",
                "status",
                "link_depth",
                "out_degree",
                "discovered_at",
            ]
        }
    elif focus == "score":
        fields = {
            k: all_fields[k]
            for k in [
                "id",
                "status",
                "interest_score",
                "score_reasoning",
                "skip_reason",
                "scored_at",
            ]
        }
    elif focus == "status":
        return json.dumps(
            {
                "status_values": status_values,
                "transitions": {
                    "crawled": ["scored", "discovered", "skipped", "failed"],
                    "scored": ["discovered", "skipped"],
                    "discovered": ["ingested", "failed", "stale"],
                    "skipped": ["discovered"],  # Can be re-evaluated
                    "ingested": ["stale"],
                    "failed": ["crawled"],  # Retry
                },
            },
            indent=2,
        )
    else:
        fields = all_fields

    return json.dumps(
        {
            "node_label": "WikiPage",
            "fields": fields,
            "relationships": {
                "LINKS_TO": {"target": "WikiPage", "direction": "outgoing"},
                "FACILITY_ID": {"target": "Facility", "direction": "outgoing"},
            },
        },
        indent=2,
    )


def get_graph_schema(
    focus: str | None = None,
    node_types: list[str] | None = None,
) -> str:
    """Get graph schema for Cypher query generation.

    Returns node labels with properties, enums, and relationship types.
    Use focus or node_types to get a subset and reduce context usage.

    Args:
        focus: Predefined schema subset:
            - "wiki": WikiPage schema for discovery/scoring
            - "facility": Facility, FacilityPath, SourceFile
            - "mdsplus": MDSplusTree, TreeNode, TDIFunction
            - "all" or None: Complete schema (default)
        node_types: Specific node labels to include (overrides focus)

    Returns:
        JSON schema with node_labels, enums, relationship_types

    Examples:
        get_graph_schema(focus="wiki")  # For wiki scoring
        get_graph_schema(node_types=["WikiPage", "Facility"])
    """
    schema = get_schema()

    # Determine which node types to include
    focus_mapping = {
        "wiki": ["WikiPage", "Facility"],
        "facility": ["Facility", "FacilityPath", "SourceFile", "AnalysisCode"],
        "mdsplus": ["MDSplusTree", "TreeNode", "TDIFunction", "Diagnostic"],
        "code": ["CodeChunk", "SourceFile", "AnalysisCode"],
    }

    if node_types:
        labels_to_include = set(node_types)
    elif focus and focus != "all":
        labels_to_include = set(focus_mapping.get(focus, schema.node_labels))
    else:
        labels_to_include = set(schema.node_labels)

    node_labels = {}
    for label in schema.node_labels:
        if label in labels_to_include:
            node_labels[label] = {
                "identifier": schema.get_identifier(label),
                "description": schema.get_class_description(label),
                "properties": schema.get_all_slots(label),
            }

    # Get relevant enums
    all_enums = schema.get_enums()

    return json.dumps(
        {
            "node_labels": node_labels,
            "enums": all_enums,
            "relationship_types": schema.relationship_types,
            "focus": focus or "all",
        },
        indent=2,
    )


# =============================================================================
# Graph Read Tools
# =============================================================================


def get_wiki_pages(
    facility_id: str,
    status: str | None = None,
    limit: int = 100,
    order_by: str = "in_degree",
) -> str:
    """Get WikiPage nodes from graph with optional filters.

    Args:
        facility_id: Facility to query (e.g., "epfl")
        status: Filter by status (crawled, discovered, skipped, etc.)
        limit: Maximum pages to return
        order_by: Sort field (in_degree, interest_score, link_depth)

    Returns:
        JSON array of page objects with id, title, status, and metrics.
    """
    with GraphClient() as gc:
        status_clause = "AND wp.status = $status" if status else ""
        order_clause = (
            f"wp.{order_by}"
            if order_by in ["in_degree", "interest_score", "link_depth", "out_degree"]
            else "wp.in_degree"
        )

        result = gc.query(
            f"""
            MATCH (wp:WikiPage {{facility_id: $facility_id}})
            WHERE true {status_clause}
            RETURN wp.id AS id,
                   wp.title AS title,
                   wp.status AS status,
                   wp.link_depth AS depth,
                   wp.in_degree AS in_degree,
                   wp.out_degree AS out_degree,
                   wp.interest_score AS score
            ORDER BY {order_clause} DESC
            LIMIT $limit
            """,
            facility_id=facility_id,
            status=status,
            limit=limit,
        )

    return json.dumps({"pages": result, "count": len(result)})


def get_wiki_neighbors(page_id: str) -> str:
    """Get pages that link to/from a specific page.

    Args:
        page_id: Full page ID (e.g., "epfl:Thomson")

    Returns:
        JSON with incoming and outgoing link info.
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (wp:WikiPage {id: $page_id})
            OPTIONAL MATCH (wp)-[:LINKS_TO]->(outgoing)
            OPTIONAL MATCH (incoming)-[:LINKS_TO]->(wp)
            WITH wp,
                 collect(DISTINCT {id: outgoing.id, title: outgoing.title, score: outgoing.interest_score})[..20] AS out_links,
                 collect(DISTINCT {id: incoming.id, title: incoming.title, score: incoming.interest_score})[..20] AS in_links
            RETURN out_links, in_links
            """,
            page_id=page_id,
        )

    if result:
        return json.dumps(result[0])
    return json.dumps({"out_links": [], "in_links": []})


def get_wiki_progress(facility_id: str) -> str:
    """Get discovery progress for a facility.

    Returns status counts, scoring progress, and recommendations.
    """
    with GraphClient() as gc:
        # Status distribution
        status_result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility_id})
            RETURN wp.status AS status, count(*) AS count
            """,
            facility_id=facility_id,
        )

        # Score distribution
        score_result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility_id})
            WHERE wp.interest_score IS NOT NULL
            RETURN
                count(CASE WHEN wp.interest_score >= 0.7 THEN 1 END) AS high_score,
                count(CASE WHEN wp.interest_score >= 0.4 AND wp.interest_score < 0.7 THEN 1 END) AS medium_score,
                count(CASE WHEN wp.interest_score < 0.4 THEN 1 END) AS low_score
            """,
            facility_id=facility_id,
        )

    status_counts = {r["status"]: r["count"] for r in status_result}
    scores = score_result[0] if score_result else {}

    return json.dumps(
        {
            "facility_id": facility_id,
            "status_counts": status_counts,
            "total_pages": sum(status_counts.values()),
            "score_distribution": {
                "high": scores.get("high_score", 0),
                "medium": scores.get("medium_score", 0),
                "low": scores.get("low_score", 0),
            },
            "recommendations": {
                "needs_scoring": status_counts.get("crawled", 0),
                "ready_for_ingest": status_counts.get("discovered", 0),
            },
        }
    )


# =============================================================================
# Graph Write Tools (UNWIND for performance)
# =============================================================================


def update_wiki_scores(scores_json: str) -> str:
    """Batch update interest_score for WikiPage nodes.

    Args:
        scores_json: JSON array of objects with:
            - id: Page ID (required)
            - score: Interest score 0.0-1.0 (required)
            - reasoning: Explanation for score (optional)
            - skip_reason: Why skipped if score < 0.5 (optional)

    Returns:
        JSON with updated count and any errors.

    Example:
        update_wiki_scores('[
            {"id": "epfl:Thomson", "score": 0.95, "reasoning": "in_degree=47, core diagnostic"},
            {"id": "epfl:Meeting_2024", "score": 0.1, "reasoning": "in_degree=0", "skip_reason": "administrative"}
        ]')
    """
    try:
        scores = json.loads(scores_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}", "updated": 0})

    if not scores:
        return json.dumps({"error": "Empty scores array", "updated": 0})

    # Prepare batch data
    batch = []
    for s in scores:
        page_id = s.get("id")
        score = s.get("score", 0.5)
        if page_id is None:
            continue

        batch.append(
            {
                "id": page_id,
                "score": score,
                "status": "discovered" if score >= 0.5 else "skipped",
                "reasoning": s.get("reasoning", ""),
                "skip_reason": s.get("skip_reason"),
            }
        )

    if not batch:
        return json.dumps({"error": "No valid entries", "updated": 0})

    # UNWIND for batch update
    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $batch AS entry
            MATCH (wp:WikiPage {id: entry.id})
            SET wp.interest_score = entry.score,
                wp.status = entry.status,
                wp.score_reasoning = entry.reasoning,
                wp.skip_reason = entry.skip_reason,
                wp.scored_at = datetime()
            """,
            batch=batch,
        )

    return json.dumps({"updated": len(batch)})


def create_wiki_pages(pages_json: str) -> str:
    """Batch create WikiPage nodes with LINKS_TO relationships.

    Args:
        pages_json: JSON array of objects with:
            - id: Page ID (required, e.g., "epfl:Thomson")
            - facility_id: Facility (required)
            - url: Wiki URL (required)
            - title: Page title (required)
            - link_depth: Distance from portal (optional)
            - out_degree: Number of outgoing links (optional)
            - links_to: Array of page IDs this page links to (optional)

    Returns:
        JSON with created count.
    """
    try:
        pages = json.loads(pages_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}", "created": 0})

    if not pages:
        return json.dumps({"error": "Empty pages array", "created": 0})

    # Create nodes with UNWIND
    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $pages AS page
            MERGE (wp:WikiPage {id: page.id})
            SET wp.facility_id = page.facility_id,
                wp.url = page.url,
                wp.title = page.title,
                wp.status = 'crawled',
                wp.link_depth = page.link_depth,
                wp.out_degree = page.out_degree,
                wp.discovered_at = datetime()
            WITH wp, page
            MATCH (f:Facility {id: page.facility_id})
            MERGE (wp)-[:FACILITY_ID]->(f)
            """,
            pages=pages,
        )

        # Create LINKS_TO relationships separately
        links_batch = []
        for p in pages:
            source_id = p.get("id")
            links_to = p.get("links_to", [])
            for target_id in links_to:
                links_batch.append({"source": source_id, "target": target_id})

        if links_batch:
            gc.query(
                """
                UNWIND $links AS link
                MATCH (source:WikiPage {id: link.source})
                MATCH (target:WikiPage {id: link.target})
                MERGE (source)-[:LINKS_TO]->(target)
                """,
                links=links_batch,
            )

    return json.dumps({"created": len(pages), "links_created": len(links_batch)})


# =============================================================================
# SSH Tools for Wiki Content
# =============================================================================


def fetch_wiki_links(
    facility_id: str,
    page_name: str,
    base_url: str = "https://spcwiki.epfl.ch/wiki",
) -> str:
    """Extract internal wiki links from a page via SSH.

    Args:
        facility_id: Facility for SSH host (e.g., "epfl")
        page_name: Page to fetch links from
        base_url: Wiki base URL

    Returns:
        JSON array of linked page names.
    """
    encoded = urllib.parse.quote(page_name, safe="")
    url = f"{base_url}/{encoded}"

    cmd = f'''curl -sk "{url}" | grep -oP 'href="/wiki/[^"#:]+' | sed 's|href="/wiki/||' | sort -u'''

    try:
        result = subprocess.run(
            ["ssh", facility_id, cmd],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return json.dumps({"links": [], "error": result.stderr})

        # Filter junk
        excluded_prefixes = (
            "Special:",
            "File:",
            "Talk:",
            "User_talk:",
            "Template:",
            "Category:",
            "Help:",
            "MediaWiki:",
            "index.php",
            "skins/",
            "opensearch",
        )
        excluded_extensions = (".css", ".js", ".php", ".png", ".jpg", ".gif")

        links = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            if line.startswith(excluded_prefixes):
                continue
            if any(line.endswith(ext) for ext in excluded_extensions):
                continue
            if "?" in line or "&" in line:
                continue
            links.append(urllib.parse.unquote(line))

        return json.dumps({"links": links, "count": len(links)})

    except subprocess.TimeoutExpired:
        return json.dumps({"links": [], "error": "Timeout"})


def fetch_wiki_preview(
    facility_id: str,
    page_name: str,
    max_chars: int = 1000,
    base_url: str = "https://spcwiki.epfl.ch/wiki",
) -> str:
    """Fetch first N characters of wiki page content.

    Args:
        facility_id: Facility for SSH host
        page_name: Page to preview
        max_chars: Maximum characters to return
        base_url: Wiki base URL

    Returns:
        JSON with title, preview text, and detected patterns.
    """
    encoded = urllib.parse.quote(page_name, safe="")
    url = f"{base_url}/{encoded}"

    # Extract title and body text
    cmd = (
        f'''curl -sk "{url}" | grep -oP '(?<=<title>)[^<]+|(?<=<p>)[^<]+' | head -20'''
    )

    try:
        result = subprocess.run(
            ["ssh", facility_id, cmd],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return json.dumps({"preview": "", "error": result.stderr})

        lines = result.stdout.strip().split("\n")
        title = lines[0] if lines else page_name
        preview = " ".join(lines[1:])[:max_chars] if len(lines) > 1 else ""

        return json.dumps(
            {
                "title": title,
                "preview": preview,
                "size": len(result.stdout),
            }
        )

    except subprocess.TimeoutExpired:
        return json.dumps({"preview": "", "error": "Timeout"})


# =============================================================================
# LlamaIndex Tool Wrappers
# =============================================================================


def get_wiki_tools() -> list[FunctionTool]:
    """Get all wiki tools as LlamaIndex FunctionTools."""
    return [
        FunctionTool.from_defaults(
            fn=get_graph_schema,
            name="get_graph_schema",
            description="Get schema for graph writes. focus='wiki'|'facility'|'mdsplus' or node_types=['WikiPage']",
        ),
        FunctionTool.from_defaults(
            fn=get_wiki_schema,
            name="get_wiki_schema",
            description="Get WikiPage schema for graph writes. Focus: 'create', 'score', 'status', or 'all'",
        ),
        FunctionTool.from_defaults(
            fn=get_wiki_pages,
            name="get_wiki_pages",
            description="Get WikiPage nodes from graph. Filter by status, order by metrics.",
        ),
        FunctionTool.from_defaults(
            fn=get_wiki_neighbors,
            name="get_wiki_neighbors",
            description="Get pages linking to/from a page. Use to assess value from context.",
        ),
        FunctionTool.from_defaults(
            fn=get_wiki_progress,
            name="get_wiki_progress",
            description="Get discovery progress: status counts, scores, recommendations.",
        ),
        FunctionTool.from_defaults(
            fn=update_wiki_scores,
            name="update_wiki_scores",
            description="Batch update scores. JSON array: [{id, score, reasoning, skip_reason}]",
        ),
        FunctionTool.from_defaults(
            fn=create_wiki_pages,
            name="create_wiki_pages",
            description="Batch create WikiPage nodes. JSON array with id, facility_id, url, title.",
        ),
        FunctionTool.from_defaults(
            fn=fetch_wiki_links,
            name="fetch_wiki_links",
            description="Extract internal links from a wiki page via SSH.",
        ),
        FunctionTool.from_defaults(
            fn=fetch_wiki_preview,
            name="fetch_wiki_preview",
            description="Fetch preview text (first 1000 chars) of a wiki page.",
        ),
    ]
