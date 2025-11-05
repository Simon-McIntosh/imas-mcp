"""Tests for IMASPythonSearchTool."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from imas_mcp.models.search_models import (
    DocsMCPServerStatus,
    IMASPythonSearchMetadata,
    IMASPythonSearchResponse,
    IMASPythonSearchResult,
    VersionInfo,
)
from imas_mcp.tools.imas_python_search_tool import IMASPythonSearchTool


@pytest.fixture
def search_tool():
    """Create an IMASPythonSearchTool instance."""
    return IMASPythonSearchTool(
        docs_mcp_url="http://localhost:3000",
        docs_db_path="./test-docs-mcp-data",
        auto_scrape_missing=True,
    )


@pytest.mark.asyncio
async def test_search_server_unavailable(search_tool):
    """Test search when docs-mcp-server is unavailable."""
    with patch.object(
        search_tool.proxy_service, "check_status"
    ) as mock_check_status:
        mock_check_status.return_value = DocsMCPServerStatus(
            available=False,
            url="http://localhost:3000",
            error="Connection refused",
        )

        response = await search_tool.search_imas_python_docs(
            query="test query",
            version="1.0.0",
        )

        assert len(response.results) == 0
        assert response.error is not None
        assert "unavailable" in response.error.lower()
        assert response.setup_instructions is not None


@pytest.mark.asyncio
async def test_search_success_with_version(search_tool):
    """Test successful search with specified version."""
    with patch.object(
        search_tool.proxy_service, "check_status"
    ) as mock_check_status, patch.object(
        search_tool.proxy_service, "list_versions"
    ) as mock_list_versions, patch.object(
        search_tool.scraping_service, "ensure_version_available"
    ) as mock_ensure, patch.object(
        search_tool.proxy_service, "search"
    ) as mock_search:

        # Mock server as available
        mock_check_status.return_value = DocsMCPServerStatus(
            available=True,
            url="http://localhost:3000",
        )

        # Mock available versions
        mock_list_versions.return_value = [
            VersionInfo(
                version="1.0.0", available=True, scraped_at=None, source_url=None
            )
        ]

        # Mock version is available
        mock_ensure.return_value = True

        # Mock search results
        mock_search.return_value = IMASPythonSearchResponse(
            query="test query",
            results=[
                IMASPythonSearchResult(
                    title="Test Result",
                    url="https://example.com/test",
                    content="Test content",
                    relevance_score=0.95,
                    section="API",
                )
            ],
            metadata=IMASPythonSearchMetadata(
                version="1.0.0",
                source="https://example.com",
                search_timestamp="2025-01-01T00:00:00Z",
                total_results=1,
            ),
        )

        response = await search_tool.search_imas_python_docs(
            query="test query",
            version="1.0.0",
        )

        assert len(response.results) == 1
        assert response.results[0].title == "Test Result"
        assert response.metadata is not None


@pytest.mark.asyncio
async def test_search_auto_detect_version(search_tool):
    """Test search with auto-detected version."""
    with patch.object(
        search_tool.proxy_service, "check_status"
    ) as mock_check_status, patch.object(
        search_tool.scraping_service, "detect_imas_python_version"
    ) as mock_detect, patch.object(
        search_tool.proxy_service, "list_versions"
    ) as mock_list_versions, patch.object(
        search_tool.scraping_service, "ensure_version_available"
    ) as mock_ensure, patch.object(
        search_tool.proxy_service, "search"
    ) as mock_search:

        # Mock server as available
        mock_check_status.return_value = DocsMCPServerStatus(
            available=True,
            url="http://localhost:3000",
        )

        # Mock version detection
        mock_detect.return_value = "1.0.0"

        # Mock available versions
        mock_list_versions.return_value = [
            VersionInfo(
                version="1.0.0", available=True, scraped_at=None, source_url=None
            )
        ]

        # Mock version is available
        mock_ensure.return_value = True

        # Mock search results
        mock_search.return_value = IMASPythonSearchResponse(
            query="test query",
            results=[],
        )

        response = await search_tool.search_imas_python_docs(query="test query")

        # Verify version was auto-detected
        mock_detect.assert_called_once()
        assert response is not None


@pytest.mark.asyncio
async def test_search_version_not_available(search_tool):
    """Test search when requested version is not available."""
    from imas_mcp.models.error_extensions import VersionNotAvailableError

    with patch.object(
        search_tool.proxy_service, "check_status"
    ) as mock_check_status, patch.object(
        search_tool.proxy_service, "list_versions"
    ) as mock_list_versions, patch.object(
        search_tool.scraping_service, "ensure_version_available"
    ) as mock_ensure:

        # Mock server as available
        mock_check_status.return_value = DocsMCPServerStatus(
            available=True,
            url="http://localhost:3000",
        )

        # Mock available versions
        mock_list_versions.return_value = [
            VersionInfo(
                version="2.0.0", available=True, scraped_at=None, source_url=None
            )
        ]

        # Mock version not available
        mock_ensure.side_effect = VersionNotAvailableError(
            version="1.0.0", available_versions=["2.0.0"]
        )

        response = await search_tool.search_imas_python_docs(
            query="test query",
            version="1.0.0",
        )

        assert len(response.results) == 0
        assert response.error is not None
        assert "1.0.0" in response.error
        assert response.setup_instructions is not None


@pytest.mark.asyncio
async def test_close(search_tool):
    """Test closing the tool."""
    with patch.object(search_tool.proxy_service, "close") as mock_close:
        mock_close.return_value = AsyncMock()

        await search_tool.close()

        mock_close.assert_called_once()
