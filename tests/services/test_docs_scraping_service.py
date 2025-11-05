"""Tests for DocsScrapingService."""

import pytest
from unittest.mock import MagicMock, patch

from imas_mcp.models.error_extensions import ScrapingFailedError, VersionNotAvailableError
from imas_mcp.models.search_models import VersionInfo
from imas_mcp.services.docs_scraping_service import DocsScrapingService


@pytest.fixture
def scraping_service():
    """Create a DocsScrapingService instance."""
    return DocsScrapingService(
        docs_db_path="./test-docs-mcp-data",
        scraping_timeout=60.0,
        auto_scrape_missing=True,
    )


@pytest.mark.asyncio
async def test_detect_imas_python_version_success(scraping_service):
    """Test successful version detection."""
    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "1.0.0\n"
        mock_run.return_value = mock_result

        version = await scraping_service.detect_imas_python_version()

        assert version == "1.0.0"


@pytest.mark.asyncio
async def test_detect_imas_python_version_failure(scraping_service):
    """Test version detection when imas is not installed."""
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError()

        version = await scraping_service.detect_imas_python_version()

        assert version is None


@pytest.mark.asyncio
async def test_scrape_version_success(scraping_service):
    """Test successful documentation scraping."""
    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Scraping completed"
        mock_run.return_value = mock_result

        result = await scraping_service.scrape_version("1.0.0")

        assert result["success"] is True
        assert result["version"] == "1.0.0"


@pytest.mark.asyncio
async def test_scrape_version_failure(scraping_service):
    """Test scraping failure."""
    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Error occurred"
        mock_result.stdout = ""
        mock_run.return_value = mock_result

        with pytest.raises(ScrapingFailedError):
            await scraping_service.scrape_version("1.0.0")


@pytest.mark.asyncio
async def test_scrape_version_timeout(scraping_service):
    """Test scraping timeout."""
    with patch("subprocess.run") as mock_run:
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=60.0)

        with pytest.raises(ScrapingFailedError) as exc_info:
            await scraping_service.scrape_version("1.0.0")

        assert "timed out" in str(exc_info.value)


@pytest.mark.asyncio
async def test_ensure_version_available_already_available(scraping_service):
    """Test ensure_version_available when version already exists."""
    available_versions = [
        VersionInfo(version="1.0.0", available=True, scraped_at=None, source_url=None)
    ]

    result = await scraping_service.ensure_version_available("1.0.0", available_versions)

    assert result is True


@pytest.mark.asyncio
async def test_ensure_version_available_auto_scrape(scraping_service):
    """Test ensure_version_available with auto-scraping."""
    available_versions = [
        VersionInfo(version="2.0.0", available=True, scraped_at=None, source_url=None)
    ]

    with patch.object(scraping_service, "scrape_version") as mock_scrape:
        mock_scrape.return_value = {"success": True, "version": "1.0.0"}

        result = await scraping_service.ensure_version_available(
            "1.0.0", available_versions
        )

        assert result is True
        mock_scrape.assert_called_once_with("1.0.0")


@pytest.mark.asyncio
async def test_ensure_version_available_auto_scrape_disabled(scraping_service):
    """Test ensure_version_available with auto-scraping disabled."""
    scraping_service.auto_scrape_missing = False
    available_versions = [
        VersionInfo(version="2.0.0", available=True, scraped_at=None, source_url=None)
    ]

    with pytest.raises(VersionNotAvailableError):
        await scraping_service.ensure_version_available("1.0.0", available_versions)


def test_get_scraping_instructions(scraping_service):
    """Test get_scraping_instructions."""
    instructions = scraping_service.get_scraping_instructions("1.0.0")

    assert "1.0.0" in instructions
    assert "npx @modelcontextprotocol/server-docs" in instructions
    assert scraping_service.docs_db_path in instructions
