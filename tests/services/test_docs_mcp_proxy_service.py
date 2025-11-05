"""Tests for DocsMCPProxyService."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from imas_mcp.models.error_extensions import DocsMCPServerUnavailableError
from imas_mcp.models.search_models import (
    DocsMCPServerStatus,
    IMASPythonSearchRequest,
    IMASPythonSearchResponse,
    VersionInfo,
)
from imas_mcp.services.docs_mcp_proxy_service import DocsMCPProxyService


@pytest.fixture
def proxy_service():
    """Create a DocsMCPProxyService instance."""
    return DocsMCPProxyService(server_url="http://localhost:3000", timeout=10.0)


@pytest.mark.asyncio
async def test_check_status_available(proxy_service):
    """Test check_status when server is available."""
    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.closed = False
        mock_session_class.return_value = mock_session

        # Set the session
        proxy_service._session = mock_session

        status = await proxy_service.check_status()

        assert status.available is True
        assert status.url == "http://localhost:3000"
        assert status.error is None


@pytest.mark.asyncio
async def test_check_status_unavailable(proxy_service):
    """Test check_status when server is unavailable."""
    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=Exception("Connection refused"))
        mock_session.closed = False
        mock_session_class.return_value = mock_session

        # Set the session
        proxy_service._session = mock_session

        status = await proxy_service.check_status()

        assert status.available is False
        assert status.error is not None


@pytest.mark.asyncio
async def test_list_versions_success(proxy_service):
    """Test list_versions with successful response."""
    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "result": {
                    "data": [
                        {
                            "version": "1.0.0",
                            "available": True,
                            "scraped_at": "2025-01-01T00:00:00Z",
                            "source_url": "https://example.com",
                        }
                    ]
                }
            }
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.closed = False
        mock_session_class.return_value = mock_session

        # Set the session
        proxy_service._session = mock_session

        versions = await proxy_service.list_versions()

        assert len(versions) == 1
        assert versions[0].version == "1.0.0"
        assert versions[0].available is True


@pytest.mark.asyncio
async def test_list_versions_failure(proxy_service):
    """Test list_versions when server fails."""
    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=Exception("Connection refused"))
        mock_session.closed = False
        mock_session_class.return_value = mock_session

        # Set the session
        proxy_service._session = mock_session

        with pytest.raises(DocsMCPServerUnavailableError):
            await proxy_service.list_versions()


@pytest.mark.asyncio
async def test_search_success(proxy_service):
    """Test search with successful response."""
    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "result": {
                    "data": {
                        "results": [
                            {
                                "title": "Test Result",
                                "url": "https://example.com/test",
                                "content": "Test content",
                                "relevance_score": 0.95,
                                "section": "API",
                            }
                        ],
                        "metadata": {
                            "version": "1.0.0",
                            "source": "https://example.com",
                            "search_timestamp": "2025-01-01T00:00:00Z",
                            "total_results": 1,
                        },
                        "suggestions": [],
                    }
                }
            }
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.closed = False
        mock_session_class.return_value = mock_session

        # Set the session
        proxy_service._session = mock_session

        request = IMASPythonSearchRequest(
            query="test query", version="1.0.0", max_results=10
        )

        response = await proxy_service.search(request)

        assert len(response.results) == 1
        assert response.results[0].title == "Test Result"
        assert response.metadata is not None
        assert response.metadata.version == "1.0.0"


@pytest.mark.asyncio
async def test_search_failure(proxy_service):
    """Test search when connection fails."""
    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=Exception("Connection refused"))
        mock_session.closed = False
        mock_session_class.return_value = mock_session

        # Set the session
        proxy_service._session = mock_session

        request = IMASPythonSearchRequest(
            query="test query", version="1.0.0", max_results=10
        )

        response = await proxy_service.search(request)

        assert len(response.results) == 0
        assert response.error is not None
        assert response.setup_instructions is not None


@pytest.mark.asyncio
async def test_close_session(proxy_service):
    """Test closing the session."""
    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        mock_session_class.return_value = mock_session

        proxy_service._session = mock_session

        await proxy_service.close()

        mock_session.close.assert_called_once()
