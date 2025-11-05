"""
Docs Scraping Service for managing IMAS-Python documentation scraping.

This service handles automatic version detection and scraping of
IMAS-Python documentation into docs-mcp-server.
"""

import logging
import subprocess
from typing import Any

from imas_mcp.models.error_extensions import ScrapingFailedError, VersionNotAvailableError
from imas_mcp.models.search_models import VersionInfo
from imas_mcp.services.base import BaseService

logger = logging.getLogger(__name__)


class DocsScrapingService(BaseService):
    """Service for managing IMAS-Python documentation scraping."""

    def __init__(
        self,
        docs_db_path: str = "./docs-mcp-data",
        scraping_timeout: float = 300.0,
        auto_scrape_missing: bool = True,
    ):
        """
        Initialize the scraping service.

        Args:
            docs_db_path: Path to store scraped documentation
            scraping_timeout: Timeout for scraping operations (seconds)
            auto_scrape_missing: Whether to automatically scrape missing versions
        """
        super().__init__()
        self.docs_db_path = docs_db_path
        self.scraping_timeout = scraping_timeout
        self.auto_scrape_missing = auto_scrape_missing

    async def detect_imas_python_version(self) -> str | None:
        """
        Detect the installed IMAS-Python version.

        Returns:
            Version string or None if not detected
        """
        try:
            # Try to import imas to get version
            result = subprocess.run(
                ["python", "-c", "import imas; print(imas.__version__)"],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                logger.info(f"Detected IMAS-Python version: {version}")
                return version
        except subprocess.TimeoutExpired:
            logger.warning("Timeout detecting IMAS-Python version")
        except Exception as e:
            logger.warning(f"Could not detect IMAS-Python version: {e}")

        return None

    async def scrape_version(
        self, version: str, source_url: str | None = None
    ) -> dict[str, Any]:
        """
        Scrape IMAS-Python documentation for a specific version.

        Args:
            version: Version to scrape
            source_url: Optional source URL (auto-detected if None)

        Returns:
            Dict with scraping results

        Raises:
            ScrapingFailedError: If scraping fails
        """
        if source_url is None:
            # Auto-detect source URL based on version
            source_url = f"https://imas-python.readthedocs.io/en/{version}/"

        logger.info(f"Scraping IMAS-Python docs version {version} from {source_url}")

        try:
            # Use docs-mcp-server CLI to scrape
            cmd = [
                "npx",
                "@modelcontextprotocol/server-docs",
                "scrape",
                "--url",
                source_url,
                "--version",
                version,
                "--output",
                self.docs_db_path,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.scraping_timeout,
            )

            if result.returncode != 0:
                raise ScrapingFailedError(
                    f"Scraping failed with exit code {result.returncode}",
                    version=version,
                    context={"stderr": result.stderr, "stdout": result.stdout},
                )

            logger.info(f"Successfully scraped IMAS-Python docs version {version}")

            return {
                "success": True,
                "version": version,
                "source_url": source_url,
                "output": result.stdout,
            }

        except subprocess.TimeoutExpired:
            raise ScrapingFailedError(
                f"Scraping timed out after {self.scraping_timeout} seconds",
                version=version,
            )
        except FileNotFoundError:
            raise ScrapingFailedError(
                "docs-mcp-server CLI not found. Install with: npm install -g @modelcontextprotocol/server-docs",
                version=version,
            )
        except Exception as e:
            raise ScrapingFailedError(
                f"Scraping failed: {str(e)}", version=version
            )

    async def ensure_version_available(
        self, version: str, available_versions: list[VersionInfo]
    ) -> bool:
        """
        Ensure a version is available, scraping if necessary.

        Args:
            version: Version to ensure is available
            available_versions: List of currently available versions

        Returns:
            True if version is available or was successfully scraped

        Raises:
            VersionNotAvailableError: If version cannot be made available
        """
        # Check if already available
        for v in available_versions:
            if v.version == version and v.available:
                logger.debug(f"Version {version} already available")
                return True

        # Auto-scrape if enabled
        if self.auto_scrape_missing:
            logger.info(f"Auto-scraping missing version {version}")
            try:
                await self.scrape_version(version)
                return True
            except ScrapingFailedError as e:
                logger.error(f"Auto-scraping failed: {e}")
                raise VersionNotAvailableError(
                    version=version,
                    available_versions=[v.version for v in available_versions],
                    context={"scraping_error": str(e)},
                )

        # Not available and auto-scrape disabled
        raise VersionNotAvailableError(
            version=version,
            available_versions=[v.version for v in available_versions],
        )

    def get_scraping_instructions(self, version: str) -> str:
        """
        Generate instructions for manual scraping.

        Args:
            version: Version to scrape

        Returns:
            Instruction text
        """
        return f"""
To manually scrape IMAS-Python documentation for version {version}:

1. Ensure docs-mcp-server is installed:
   npm install -g @modelcontextprotocol/server-docs

2. Run the scraping command:
   python scripts/scrape_imas_docs.py --version {version}

3. Or use the docs-mcp-server CLI directly:
   npx @modelcontextprotocol/server-docs scrape \\
     --url https://imas-python.readthedocs.io/en/{version}/ \\
     --version {version} \\
     --output {self.docs_db_path}

4. Restart the docs-mcp-server to load the new documentation.
"""
