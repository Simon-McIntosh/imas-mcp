"""
IMAS Python Search Tool for searching IMAS-Python documentation.

This tool integrates with docs-mcp-server to provide search functionality
for IMAS-Python API documentation, tutorials, and guides.
"""

import logging
from typing import Any

from fastmcp import Context

from imas_mcp.models.error_extensions import (
    DocsMCPServerUnavailableError,
    VersionNotAvailableError,
)
from imas_mcp.models.search_models import (
    IMASPythonSearchRequest,
    IMASPythonSearchResponse,
)
from imas_mcp.search.decorators import handle_errors, mcp_tool, measure_performance
from imas_mcp.services.docs_mcp_proxy_service import DocsMCPProxyService
from imas_mcp.services.docs_scraping_service import DocsScrapingService

logger = logging.getLogger(__name__)


class IMASPythonSearchTool:
    """Tool for searching IMAS-Python documentation."""

    def __init__(
        self,
        docs_mcp_url: str = "http://localhost:3000",
        docs_db_path: str = "./docs-mcp-data",
        auto_scrape_missing: bool = True,
        timeout: float = 30.0,
    ):
        """
        Initialize the IMAS-Python search tool.

        Args:
            docs_mcp_url: URL of the docs-mcp-server
            docs_db_path: Path to documentation database
            auto_scrape_missing: Whether to auto-scrape missing versions
            timeout: Request timeout in seconds
        """
        self.logger = logger
        self.tool_name = "search_imas_python_docs"

        # Initialize services
        self.proxy_service = DocsMCPProxyService(
            server_url=docs_mcp_url,
            timeout=timeout,
        )
        self.scraping_service = DocsScrapingService(
            docs_db_path=docs_db_path,
            auto_scrape_missing=auto_scrape_missing,
        )

    @handle_errors(fallback="search_suggestions")
    @measure_performance(include_metrics=True, slow_threshold=2.0)
    @mcp_tool(
        "Search IMAS-Python documentation including API references, tutorials, and guides. "
        "Automatically detects IMAS-Python version and scrapes documentation if needed."
    )
    async def search_imas_python_docs(
        self,
        query: str,
        version: str | None = None,
        max_results: int = 10,
        ctx: Context | None = None,
    ) -> IMASPythonSearchResponse:
        """
        Search IMAS-Python documentation for API references, tutorials, and guides.

        This tool provides access to the official IMAS-Python documentation,
        helping users find information about:
        - Python API functions and classes
        - Data access patterns and workflows
        - Installation and configuration
        - Tutorials and examples
        - Best practices

        Args:
            query: Search query (e.g., "open IDS file", "get core_profiles", "plot equilibrium")
            version: Specific IMAS-Python version (auto-detected if not provided)
            max_results: Maximum number of documentation pages to return (1-50)
            ctx: FastMCP context (optional)

        Returns:
            IMASPythonSearchResponse with relevant documentation pages

        Note:
            Requires docs-mcp-server to be running. If the server is unavailable,
            response will include setup instructions.
        """
        self.logger.info(
            f"Searching IMAS-Python docs: query='{query}', version={version}"
        )

        # Check if docs-mcp-server is available
        status = await self.proxy_service.check_status()
        if not status.available:
            self.logger.warning(
                f"docs-mcp-server unavailable: {status.error}"
            )
            return IMASPythonSearchResponse(
                query=query,
                results=[],
                error=f"docs-mcp-server unavailable: {status.error}",
                setup_instructions=self.proxy_service._get_setup_instructions(),
                suggestions=[
                    "Start docs-mcp-server with: python scripts/start_docs_server.py",
                    "Or run manually: npx @modelcontextprotocol/server-docs --port 3000",
                    "For Docker: ensure docker-compose.yml includes docs-mcp-server service",
                ],
            )

        # Auto-detect version if not provided
        if version is None:
            version = await self.scraping_service.detect_imas_python_version()
            if version:
                self.logger.info(f"Auto-detected IMAS-Python version: {version}")
            else:
                # Use latest or default version
                version = "latest"
                self.logger.info("Using default version: latest")

        # Check if version is available
        try:
            available_versions = await self.proxy_service.list_versions()

            # Ensure version is available (may trigger auto-scraping)
            await self.scraping_service.ensure_version_available(
                version, available_versions
            )
        except DocsMCPServerUnavailableError as e:
            return IMASPythonSearchResponse(
                query=query,
                results=[],
                error=str(e),
                setup_instructions=e.setup_instructions,
                suggestions=[
                    "Check that docs-mcp-server is running",
                    "Verify the server URL is correct",
                    "Review Docker configuration if using containers",
                ],
            )
        except VersionNotAvailableError as e:
            return IMASPythonSearchResponse(
                query=query,
                results=[],
                error=str(e),
                setup_instructions=self.scraping_service.get_scraping_instructions(
                    version
                ),
                suggestions=[
                    f"Scrape the documentation: python scripts/scrape_imas_docs.py --version {version}",
                    f"Use an available version: {', '.join(e.available_versions[:3])}",
                    "Or use 'latest' for the most recent version",
                ],
            )

        # Perform search
        request = IMASPythonSearchRequest(
            query=query,
            version=version,
            max_results=max_results,
        )

        response = await self.proxy_service.search(request)

        self.logger.info(
            f"IMAS-Python search completed: {len(response.results)} results"
        )

        return response

    async def close(self) -> None:
        """Close the tool and cleanup resources."""
        await self.proxy_service.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
