#!/usr/bin/env python3
"""
Start docs-mcp-server for local development.

This script starts the docs-mcp-server on localhost:3000 for local
development and testing of IMAS-Python documentation search.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_npm_installed() -> bool:
    """Check if npm is installed."""
    try:
        subprocess.run(
            ["npm", "--version"],
            capture_output=True,
            check=True,
            timeout=5.0,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_docs_mcp_server_installed() -> bool:
    """Check if docs-mcp-server is installed."""
    try:
        result = subprocess.run(
            ["npx", "@modelcontextprotocol/server-docs", "--version"],
            capture_output=True,
            timeout=10.0,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def install_docs_mcp_server() -> bool:
    """Install docs-mcp-server globally."""
    logger.info("Installing @modelcontextprotocol/server-docs...")
    try:
        subprocess.run(
            ["npm", "install", "-g", "@modelcontextprotocol/server-docs"],
            check=True,
            timeout=120.0,
        )
        logger.info("✓ docs-mcp-server installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install docs-mcp-server: {e}")
        return False
    except subprocess.TimeoutExpired:
        logger.error("Installation timed out")
        return False


def start_server(port: int = 3000, db_path: str = "./docs-mcp-data") -> None:
    """Start the docs-mcp-server."""
    # Ensure db directory exists
    Path(db_path).mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting docs-mcp-server on port {port}...")
    logger.info(f"Database path: {db_path}")
    logger.info("Press Ctrl+C to stop the server")

    try:
        subprocess.run(
            [
                "npx",
                "@modelcontextprotocol/server-docs",
                "--port",
                str(port),
                "--db",
                db_path,
            ],
            check=True,
        )
    except KeyboardInterrupt:
        logger.info("\nStopping docs-mcp-server...")
    except subprocess.CalledProcessError as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Start docs-mcp-server for local development"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3000,
        help="Port to run the server on (default: 3000)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="./docs-mcp-data",
        help="Path to documentation database (default: ./docs-mcp-data)",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install docs-mcp-server if not already installed",
    )

    args = parser.parse_args()

    # Check if npm is installed
    if not check_npm_installed():
        logger.error("npm is not installed. Please install Node.js and npm first.")
        logger.error("Visit: https://nodejs.org/")
        sys.exit(1)

    # Check if docs-mcp-server is installed
    if not check_docs_mcp_server_installed():
        if args.install:
            if not install_docs_mcp_server():
                sys.exit(1)
        else:
            logger.error("docs-mcp-server is not installed.")
            logger.error("Run with --install flag to install it, or run:")
            logger.error("  npm install -g @modelcontextprotocol/server-docs")
            sys.exit(1)

    # Start the server
    start_server(port=args.port, db_path=args.db_path)


if __name__ == "__main__":
    main()
