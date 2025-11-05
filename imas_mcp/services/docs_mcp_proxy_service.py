"""
Docs MCP Proxy Service for communicating with docs-mcp-server.

This service handles all communication with the external docs-mcp-server
via tRPC API calls.
"""

import logging
from datetime import UTC, datetime
from typing import Any

import aiohttp

from imas_mcp.models.error_extensions import DocsMCPServerUnavailableError
from imas_mcp.models.search_models import (
    DocsMCPServerStatus,
    IMASPythonSearchMetadata,
    IMASPythonSearchRequest,
    IMASPythonSearchResult,
    IMASPythonSearchResponse,
    VersionInfo,
)
from imas_mcp.services.base import BaseService

logger = logging.getLogger(__name__)


class DocsMCPProxyService(BaseService):
    """Service for proxying requests to docs-mcp-server via tRPC API."""

    def __init__(
        self,
        server_url: str = "http://localhost:3000",
        timeout: float = 30.0,
    ):
        """
        Initialize the docs-mcp-server proxy.

        Args:
            server_url: Base URL of the docs-mcp-server
            timeout: Request timeout in seconds
        """
        super().__init__()
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def check_status(self) -> DocsMCPServerStatus:
        """
        Check if docs-mcp-server is available.

        Returns:
            DocsMCPServerStatus with availability information
        """
        try:
            session = await self._get_session()
            async with session.get(f"{self.server_url}/health") as response:
                if response.status == 200:
                    return DocsMCPServerStatus(available=True, url=self.server_url)
                else:
                    return DocsMCPServerStatus(
                        available=False,
                        url=self.server_url,
                        error=f"Health check returned status {response.status}",
                    )
        except aiohttp.ClientError as e:
            logger.warning(f"docs-mcp-server unavailable at {self.server_url}: {e}")
            return DocsMCPServerStatus(
                available=False,
                url=self.server_url,
                error=f"Connection failed: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Unexpected error checking docs-mcp-server status: {e}")
            return DocsMCPServerStatus(
                available=False,
                url=self.server_url,
                error=f"Unexpected error: {str(e)}",
            )

    async def list_versions(self) -> list[VersionInfo]:
        """
        List available IMAS-Python versions in docs-mcp-server.

        Returns:
            List of VersionInfo objects

        Raises:
            DocsMCPServerUnavailableError: If server is not available
        """
        try:
            session = await self._get_session()
            # tRPC query endpoint
            url = f"{self.server_url}/trpc/versions.list"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get("result", {}).get("data", [])
                    return [VersionInfo(**v) for v in result]
                else:
                    error_text = await response.text()
                    raise DocsMCPServerUnavailableError(
                        f"Failed to list versions: {response.status}",
                        context={"error_text": error_text},
                    )
        except aiohttp.ClientError as e:
            raise DocsMCPServerUnavailableError(
                f"Connection to docs-mcp-server failed: {str(e)}",
                setup_instructions=self._get_setup_instructions(),
            )
        except Exception as e:
            logger.error(f"Error listing versions: {e}")
            raise DocsMCPServerUnavailableError(
                f"Failed to list versions: {str(e)}",
                setup_instructions=self._get_setup_instructions(),
            )

    async def search(
        self, request: IMASPythonSearchRequest
    ) -> IMASPythonSearchResponse:
        """
        Search IMAS-Python documentation via docs-mcp-server.

        Args:
            request: Search request parameters

        Returns:
            IMASPythonSearchResponse with results

        Raises:
            DocsMCPServerUnavailableError: If server is not available
        """
        try:
            session = await self._get_session()
            # tRPC mutation endpoint
            url = f"{self.server_url}/trpc/search.query"

            # Build tRPC request payload
            payload = {
                "json": {
                    "query": request.query,
                    "version": request.version,
                    "maxResults": request.max_results,
                }
            }

            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get("result", {}).get("data", {})

                    # Parse response
                    results = [
                        IMASPythonSearchResult(**r) for r in result.get("results", [])
                    ]

                    metadata = None
                    if "metadata" in result:
                        metadata = IMASPythonSearchMetadata(**result["metadata"])

                    return IMASPythonSearchResponse(
                        query=request.query,
                        results=results,
                        metadata=metadata,
                        suggestions=result.get("suggestions", []),
                    )
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Search failed with status {response.status}: {error_text}"
                    )
                    return IMASPythonSearchResponse(
                        query=request.query,
                        results=[],
                        error=f"Search failed: {response.status}",
                        setup_instructions=self._get_setup_instructions(),
                    )

        except aiohttp.ClientError as e:
            logger.warning(f"Connection to docs-mcp-server failed: {e}")
            return IMASPythonSearchResponse(
                query=request.query,
                results=[],
                error=f"Connection failed: {str(e)}",
                setup_instructions=self._get_setup_instructions(),
            )
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            return IMASPythonSearchResponse(
                query=request.query,
                results=[],
                error=f"Unexpected error: {str(e)}",
                setup_instructions=self._get_setup_instructions(),
            )

    def _get_setup_instructions(self) -> str:
        """Generate setup instructions for docs-mcp-server."""
        return f"""
docs-mcp-server is not available at {self.server_url}.

To enable IMAS-Python documentation search:

1. For Docker deployment:
   - Ensure docker-compose.yml includes the docs-mcp-server service
   - Run: docker-compose up -d

2. For local development:
   - Install docs-mcp-server: npm install -g @modelcontextprotocol/server-docs
   - Start the server: python scripts/start_docs_server.py
   - Or manually: npx @modelcontextprotocol/server-docs --port 3000

3. Verify the server is running:
   - Check: curl {self.server_url}/health

For more information, see the documentation in DOCKER.md and README.md.
"""

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
