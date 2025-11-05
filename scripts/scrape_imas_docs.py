#!/usr/bin/env python3
"""
Scrape IMAS-Python documentation for docs-mcp-server.

This script scrapes IMAS-Python documentation from ReadTheDocs
and stores it in a format suitable for docs-mcp-server.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def detect_imas_python_version() -> str | None:
    """Detect the installed IMAS-Python version."""
    try:
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
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def scrape_documentation(
    version: str, output_path: str, timeout: float = 300.0
) -> bool:
    """
    Scrape IMAS-Python documentation.

    Args:
        version: Version to scrape
        output_path: Output directory path
        timeout: Scraping timeout in seconds

    Returns:
        True if successful, False otherwise
    """
    # Ensure output directory exists
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Build source URL
    source_url = f"https://imas-python.readthedocs.io/en/{version}/"

    logger.info(f"Scraping IMAS-Python documentation...")
    logger.info(f"  Version: {version}")
    logger.info(f"  Source: {source_url}")
    logger.info(f"  Output: {output_path}")

    try:
        result = subprocess.run(
            [
                "npx",
                "@modelcontextprotocol/server-docs",
                "scrape",
                "--url",
                source_url,
                "--version",
                version,
                "--output",
                output_path,
            ],
            check=True,
            timeout=timeout,
        )

        logger.info("✓ Documentation scraped successfully")
        logger.info(f"✓ Data saved to: {output_path}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Scraping failed with exit code {e.returncode}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"Scraping timed out after {timeout} seconds")
        return False
    except FileNotFoundError:
        logger.error("docs-mcp-server CLI not found")
        logger.error("Install with: npm install -g @modelcontextprotocol/server-docs")
        return False


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Scrape IMAS-Python documentation for docs-mcp-server"
    )
    parser.add_argument(
        "--version",
        type=str,
        help="IMAS-Python version to scrape (auto-detected if not specified)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./docs-mcp-data",
        help="Output directory path (default: ./docs-mcp-data)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Scraping timeout in seconds (default: 300)",
    )

    args = parser.parse_args()

    # Determine version
    version = args.version
    if version is None:
        version = detect_imas_python_version()
        if version is None:
            logger.warning("Could not detect IMAS-Python version, using 'latest'")
            version = "latest"

    # Scrape documentation
    success = scrape_documentation(
        version=version,
        output_path=args.output,
        timeout=args.timeout,
    )

    if success:
        logger.info("\nNext steps:")
        logger.info("1. Start docs-mcp-server: python scripts/start_docs_server.py")
        logger.info("2. Or restart if already running")
        sys.exit(0)
    else:
        logger.error("\nScraping failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
