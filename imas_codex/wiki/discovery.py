"""Three-phase wiki discovery pipeline.

Phase 1: CRAWL - Fast link extraction, builds wiki graph structure
Phase 2: SCORE - Agent evaluates graph metrics, assigns interest scores
Phase 3: INGEST - Fetch content for high-score pages, create chunks

This module is facility-agnostic - wiki configuration comes from facility YAML.

Example:
    from imas_codex.wiki.discovery import WikiDiscovery

    discovery = WikiDiscovery("epfl", cost_limit_usd=10.0)
    await discovery.run()
"""

import logging
import subprocess
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from imas_codex.agents.llm import get_llm, get_model_for_task
from imas_codex.agents.prompt_loader import load_prompts
from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class DiscoveryStats:
    """Statistics for discovery progress."""

    # Phase 1: Crawl
    pages_crawled: int = 0
    links_found: int = 0
    max_depth_reached: int = 0
    frontier_size: int = 0

    # Phase 2: Score
    pages_scored: int = 0
    high_score_count: int = 0  # interest_score >= 0.7
    low_score_count: int = 0  # interest_score < 0.3

    # Phase 3: Ingest
    pages_ingested: int = 0
    chunks_created: int = 0

    # Cost tracking
    cost_spent_usd: float = 0.0
    cost_limit_usd: float = 10.0

    # Timing
    start_time: float = field(default_factory=time.time)
    phase: str = "idle"

    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    def cost_remaining(self) -> float:
        return max(0, self.cost_limit_usd - self.cost_spent_usd)

    def budget_exhausted(self) -> bool:
        return self.cost_spent_usd >= self.cost_limit_usd


@dataclass
class WikiConfig:
    """Wiki configuration for a facility."""

    base_url: str
    portal_page: str
    ssh_host: str
    facility_id: str

    @classmethod
    def from_facility(cls, facility: str) -> "WikiConfig":
        """Load wiki config from facility configuration."""
        # Default configurations per facility
        configs = {
            "epfl": {
                "base_url": "https://spcwiki.epfl.ch/wiki",
                "portal_page": "Portal:TCV",
                "ssh_host": "epfl",
            },
        }

        if facility not in configs:
            raise ValueError(
                f"Unknown facility: {facility}. Known: {list(configs.keys())}"
            )

        cfg = configs[facility]
        return cls(
            base_url=cfg["base_url"],
            portal_page=cfg["portal_page"],
            ssh_host=cfg["ssh_host"],
            facility_id=facility,
        )


class WikiDiscovery:
    """Three-phase wiki discovery pipeline."""

    def __init__(
        self,
        facility: str,
        cost_limit_usd: float = 10.0,
        max_pages: int = 2000,
        max_depth: int = 10,
        verbose: bool = False,
    ):
        self.config = WikiConfig.from_facility(facility)
        self.stats = DiscoveryStats(cost_limit_usd=cost_limit_usd)
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.verbose = verbose

        # Graph client for persistence
        self._gc: GraphClient | None = None

    def _get_gc(self) -> GraphClient:
        if self._gc is None:
            self._gc = GraphClient()
        return self._gc

    # =========================================================================
    # Phase 1: CRAWL - Fast link extraction
    # =========================================================================

    def _extract_links_from_page(self, page_name: str) -> list[str]:
        """Extract all internal wiki links from a page via SSH.

        Returns list of page names (not full URLs).
        """
        encoded = urllib.parse.quote(page_name, safe="")
        url = f"{self.config.base_url}/{encoded}"

        # Extract hrefs that point to internal wiki pages
        cmd = f'''curl -sk "{url}" | grep -oP 'href="/wiki/[^"#:]+' | sed 's|href="/wiki/||' | sort -u'''

        try:
            result = subprocess.run(
                ["ssh", self.config.ssh_host, cmd],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return []

            links = []
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

            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                # Skip excluded prefixes
                if line.startswith(excluded_prefixes):
                    continue
                # Skip file extensions
                if any(line.endswith(ext) for ext in excluded_extensions):
                    continue
                # Skip query strings
                if "?" in line or "&" in line:
                    continue
                # Decode URL encoding
                decoded = urllib.parse.unquote(line)
                links.append(decoded)

            return links

        except subprocess.TimeoutExpired:
            logger.warning("Timeout extracting links from %s", page_name)
            return []

    def _crawl_batch(self, pages: list[str]) -> dict[str, list[str]]:
        """Crawl multiple pages in parallel, return {page: [links]}."""
        results = {}

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(self._extract_links_from_page, page): page
                for page in pages
            }

            for future in as_completed(futures):
                page = futures[future]
                try:
                    links = future.result()
                    results[page] = links
                except Exception as e:
                    logger.warning("Error crawling %s: %s", page, e)
                    results[page] = []

        return results

    def phase1_crawl(self, progress: Progress | None = None) -> int:
        """Phase 1: Crawl wiki and build link structure.

        Graph-driven: picks up pending_crawl pages from previous runs.
        Returns number of pages crawled.
        """
        self.stats.phase = "CRAWL"
        gc = self._get_gc()

        # Load already-crawled pages to skip
        crawled_result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility_id})
            WHERE wp.status = 'crawled' OR wp.status = 'scored'
                  OR wp.status = 'discovered' OR wp.status = 'ingested'
            RETURN wp.title AS title
            """,
            facility_id=self.config.facility_id,
        )
        visited: set[str] = {r["title"] for r in crawled_result}

        # Load pending_crawl pages from previous runs
        pending_result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility_id, status: 'pending_crawl'})
            RETURN wp.title AS title, wp.link_depth AS depth
            ORDER BY wp.link_depth
            """,
            facility_id=self.config.facility_id,
        )
        frontier: set[str] = {r["title"] for r in pending_result}
        depth_map: dict[str, int] = {
            r["title"]: r["depth"] or 0 for r in pending_result
        }

        # Add portal if no frontier
        portal = self.config.portal_page
        if not frontier and portal not in visited:
            frontier.add(portal)
            depth_map[portal] = 0

        if self.verbose and pending_result:
            console.print(f"  Resuming with {len(frontier)} pending pages from graph")

        task_id = None
        if progress:
            task_id = progress.add_task("Crawling wiki...", total=self.max_pages)

        # Track all links for bulk relationship creation
        all_link_results: dict[str, list[str]] = {}
        crawled_this_run = 0

        while frontier and crawled_this_run < self.max_pages:
            # Get next batch from frontier
            batch = list(frontier)[:50]
            frontier -= set(batch)

            # Crawl batch
            results = self._crawl_batch(batch)

            for page, links in results.items():
                if page in visited:
                    continue

                visited.add(page)
                crawled_this_run += 1
                all_link_results[page] = links
                current_depth = depth_map.get(page, 0)
                self.stats.max_depth_reached = max(
                    self.stats.max_depth_reached, current_depth
                )

                # Create WikiPage node
                page_id = f"{self.config.facility_id}:{page}"
                gc.query(
                    """
                    MERGE (wp:WikiPage {id: $id})
                    SET wp.title = $title,
                        wp.url = $url,
                        wp.status = 'crawled',
                        wp.facility_id = $facility_id,
                        wp.link_depth = $depth,
                        wp.out_degree = $out_degree,
                        wp.discovered_at = datetime()
                    WITH wp
                    MATCH (f:Facility {id: $facility_id})
                    MERGE (wp)-[:FACILITY_ID]->(f)
                    """,
                    id=page_id,
                    title=page,
                    url=f"{self.config.base_url}/{urllib.parse.quote(page, safe='')}",
                    facility_id=self.config.facility_id,
                    depth=current_depth,
                    out_degree=len(links),
                )

                # Add new links to frontier
                for link in links:
                    if link not in visited and link not in frontier:
                        if current_depth + 1 <= self.max_depth:
                            frontier.add(link)
                            depth_map[link] = current_depth + 1

                self.stats.pages_crawled += 1
                self.stats.links_found += len(links)

                if progress and task_id is not None:
                    progress.update(task_id, completed=crawled_this_run)

            self.stats.frontier_size = len(frontier)

            if self.verbose:
                console.print(
                    f"  Crawled {crawled_this_run}, frontier: {len(frontier)}, depth: {self.stats.max_depth_reached}"
                )

        # Create LINKS_TO relationships in bulk
        if all_link_results:
            self._create_link_relationships(all_link_results, gc)

        # Persist frontier pages with status='pending_crawl'
        if frontier:
            self._persist_frontier(frontier, depth_map, gc)

        # Compute in_degree for all pages
        gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility_id})
            OPTIONAL MATCH (wp)<-[:LINKS_TO]-(source)
            WITH wp, count(source) AS in_deg
            SET wp.in_degree = in_deg
        """,
            facility_id=self.config.facility_id,
        )

        if progress and task_id is not None:
            progress.update(task_id, completed=self.max_pages)

        return crawled_this_run

    def _persist_frontier(
        self, frontier: set[str], depth_map: dict[str, int], gc: GraphClient
    ) -> None:
        """Persist frontier pages as pending_crawl for subsequent runs."""
        frontier_data = [
            {
                "id": f"{self.config.facility_id}:{page}",
                "title": page,
                "url": f"{self.config.base_url}/{urllib.parse.quote(page, safe='')}",
                "facility_id": self.config.facility_id,
                "link_depth": depth_map.get(page, 0),
            }
            for page in frontier
        ]

        gc.query(
            """
            UNWIND $pages AS p
            MERGE (wp:WikiPage {id: p.id})
            ON CREATE SET
                wp.title = p.title,
                wp.url = p.url,
                wp.status = 'pending_crawl',
                wp.facility_id = p.facility_id,
                wp.link_depth = p.link_depth,
                wp.discovered_at = datetime()
            WITH wp, p
            MATCH (f:Facility {id: p.facility_id})
            MERGE (wp)-[:FACILITY_ID]->(f)
            """,
            pages=frontier_data,
        )

        if self.verbose:
            console.print(f"  Persisted {len(frontier)} frontier pages for next run")

    def _create_link_relationships(
        self, results: dict[str, list[str]], gc: GraphClient
    ) -> None:
        """Create LINKS_TO relationships in bulk."""
        for source_page, target_pages in results.items():
            if not target_pages:
                continue

            source_id = f"{self.config.facility_id}:{source_page}"
            target_ids = [f"{self.config.facility_id}:{t}" for t in target_pages]

            gc.query(
                """
                MATCH (source:WikiPage {id: $source_id})
                UNWIND $target_ids AS target_id
                MATCH (target:WikiPage {id: target_id})
                MERGE (source)-[:LINKS_TO]->(target)
                """,
                source_id=source_id,
                target_ids=target_ids,
            )

    # =========================================================================
    # Phase 2: SCORE - Agent evaluates graph metrics
    # =========================================================================

    def _get_scoring_tools(self) -> list[FunctionTool]:
        """Get tools for the scoring agent from shared wiki tools."""
        from imas_codex.wiki.tools import get_wiki_tools

        # Get shared tools
        tools = get_wiki_tools()

        # Add a facility-bound progress tracker that updates stats
        facility_id = self.config.facility_id

        def track_scoring_progress() -> str:
            """Track scoring and update stats."""
            import json

            from imas_codex.wiki.tools import get_wiki_progress

            progress_json = get_wiki_progress(facility_id)
            progress = json.loads(progress_json)

            # Update internal stats
            status_counts = progress.get("status_counts", {})
            self.stats.pages_scored = status_counts.get(
                "discovered", 0
            ) + status_counts.get("skipped", 0)
            self.stats.high_score_count = progress.get("score_distribution", {}).get(
                "high", 0
            )
            self.stats.low_score_count = progress.get("score_distribution", {}).get(
                "low", 0
            )

            return progress_json

        # Add tracking tool
        tools.append(
            FunctionTool.from_defaults(
                fn=track_scoring_progress,
                name="track_scoring_progress",
                description="Track scoring progress and update internal stats.",
            )
        )

        return tools

    async def phase2_score(self, progress: Progress | None = None) -> int:
        """Phase 2: Score pages using agent with graph metrics.

        Returns number of pages scored.
        """
        self.stats.phase = "SCORE"

        # Load prompt
        prompts = load_prompts()
        system_prompt = prompts.get("wiki-scorer")

        if system_prompt is None:
            # Use default prompt
            system_prompt_text = self._get_default_scorer_prompt()
        else:
            system_prompt_text = system_prompt.content

        # Use Sonnet for scoring - higher capability for accurate assessment
        model = get_model_for_task("scoring")  # Use scoring-specific model
        llm = get_llm(model=model, temperature=0.2, max_tokens=16384)

        tools = self._get_scoring_tools()
        agent = ReActAgent(
            tools=tools,
            llm=llm,
            verbose=self.verbose,
            system_prompt=system_prompt_text,
            max_iterations=50,  # More iterations for larger batches
        )

        # Run agent with task that uses get_graph_schema for schema info
        task = f"""Score all crawled wiki pages for {self.config.facility_id}.

1. First call get_graph_schema(focus="wiki") to see WikiPage fields and valid status values
2. Call get_wiki_pages("{self.config.facility_id}", status="crawled", limit=750) to get pages
3. For each page, compute interest_score (0.0-1.0) based on graph metrics:
   - in_degree: Pages with many incoming links are important
   - out_degree: Hub pages that link to many others
   - link_depth: Pages closer to portal (lower depth) are more central
   - title: Keywords like Thomson, LIUQE, signals indicate high value
4. If uncertain about a page, call get_wiki_neighbors(page_id) to check context
5. Call update_wiki_scores with JSON array of 200-500 pages per batch
6. Call track_scoring_progress to check remaining work
7. Continue until all crawled pages are scored"""

        try:
            response = await agent.run(task)
            if self.verbose and hasattr(response, "response"):
                resp_text = str(response.response)[:200] if response.response else ""
                console.print(f"[dim]Agent response: {resp_text}...[/dim]")
        except Exception as e:
            logger.error("Scoring agent error: %s", e)

        return self.stats.pages_scored

    def _get_default_scorer_prompt(self) -> str:
        """Default system prompt for scoring agent."""
        return """You are scoring wiki pages for a fusion research facility.

Your goal is to assign interest_score (0.0-1.0) to each page based on measurable graph metrics.

## Scoring Guidelines

HIGH SCORE (0.7-1.0):
- in_degree > 5: Many pages link here - indicates importance
- Title contains: Thomson, CXRS, LIUQE, signals, nodes, calibration
- link_depth <= 2: Close to portal, central to documentation

MEDIUM SCORE (0.4-0.7):
- in_degree 1-5: Some references
- Technical content but not central
- link_depth 3-4

LOW SCORE (0.0-0.4):
- in_degree = 0: Orphan page, nobody references it
- Title contains: Meeting, Workshop, Mission, personal
- link_depth > 5: Far from main documentation

## Important

- ALWAYS provide reasoning for scores
- Skip pages with skip_reason if score < 0.5
- Use neighbor_info to check context when title is ambiguous
- Process in batches of 200-500 pages for efficiency
- Stop when all pages scored or budget exhausted"""

    # =========================================================================
    # Phase 3: INGEST - Not implemented yet (placeholder)
    # =========================================================================

    async def phase3_ingest(self, progress: Progress | None = None) -> int:
        """Phase 3: Fetch and ingest high-score pages.

        Returns number of pages ingested.
        """
        self.stats.phase = "INGEST"
        # TODO: Implement full content fetching and chunking
        # For now, just return 0
        return 0

    # =========================================================================
    # Main Entry Point
    # =========================================================================

    async def run(self) -> DiscoveryStats:
        """Run full three-phase discovery pipeline."""
        console.print(f"[bold]Wiki Discovery: {self.config.facility_id}[/bold]")
        console.print(f"Portal: {self.config.portal_page}")
        console.print(f"Cost limit: ${self.stats.cost_limit_usd:.2f}")
        console.print()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            # Phase 1: Crawl
            console.print("[cyan]Phase 1: CRAWL[/cyan]")
            crawled = self.phase1_crawl(progress)
            console.print(
                f"  Crawled {crawled} pages, {self.stats.links_found} links, depth {self.stats.max_depth_reached}"
            )

            # Phase 2: Score
            console.print("\n[cyan]Phase 2: SCORE[/cyan]")
            scored = await self.phase2_score(progress)
            console.print(
                f"  Scored {scored} pages: {self.stats.high_score_count} high, {self.stats.low_score_count} low"
            )

            # Phase 3: Ingest (placeholder)
            # console.print("\n[cyan]Phase 3: INGEST[/cyan]")
            # ingested = await self.phase3_ingest(progress)
            # console.print(f"  Ingested {ingested} pages")

        console.print(
            f"\n[green]Discovery complete in {self.stats.elapsed_seconds():.1f}s[/green]"
        )
        console.print(f"Cost: ${self.stats.cost_spent_usd:.4f}")

        return self.stats

    def close(self) -> None:
        """Close graph connection."""
        if self._gc:
            self._gc.close()
            self._gc = None


async def run_wiki_discovery(
    facility: str = "epfl",
    cost_limit_usd: float = 10.0,
    max_pages: int = 2000,
    max_depth: int = 10,
    verbose: bool = False,
) -> dict:
    """Run wiki discovery and return stats as dict.

    Args:
        facility: Facility ID (e.g., "epfl")
        cost_limit_usd: Maximum cost budget
        max_pages: Maximum pages to crawl
        max_depth: Maximum link depth from portal
        verbose: Enable verbose output

    Returns:
        Dictionary with discovery statistics
    """
    discovery = WikiDiscovery(
        facility=facility,
        cost_limit_usd=cost_limit_usd,
        max_pages=max_pages,
        max_depth=max_depth,
        verbose=verbose,
    )

    try:
        stats = await discovery.run()
        return {
            "pages_crawled": stats.pages_crawled,
            "links_found": stats.links_found,
            "pages_scored": stats.pages_scored,
            "high_score_count": stats.high_score_count,
            "low_score_count": stats.low_score_count,
            "pages_ingested": stats.pages_ingested,
            "cost_spent_usd": stats.cost_spent_usd,
            "elapsed_seconds": stats.elapsed_seconds(),
        }
    finally:
        discovery.close()
