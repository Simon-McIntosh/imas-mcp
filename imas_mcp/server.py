"""
IMAS MCP Server - Composable Integrator.

This is the principal MCP server for the IMAS data dictionary that uses
composition to combine tools and resources from separate providers.
This architecture enables clean separation of concerns and better maintainability.

The server integrates:
- Tools: 8 core tools for physics-based search and analysis
- Resources: Static JSON schema resources for reference data

Each component is accessible via server.tools and server.resources properties.
"""

import importlib.metadata
import logging
from dataclasses import dataclass, field
from typing import Optional

import nest_asyncio
from fastmcp import FastMCP

from imas_mcp.resources import Resources
from imas_mcp.tools import Tools

# apply nest_asyncio to allow nested event loops
# This is necessary for Jupyter notebooks and some other environments
# that don't support nested event loops by default.
nest_asyncio.apply()

# Configure logging with specific control over different components
logging.basicConfig(
    level=logging.WARNING, format="%(name)s - %(levelname)s - %(message)s"
)

# Set our application logger to INFO for useful messages
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Suppress FastMCP startup messages by setting to ERROR level
# This prevents the "Starting MCP server" message from appearing as a warning
fastmcp_server_logger = logging.getLogger("FastMCP.fastmcp.server.server")
fastmcp_server_logger.setLevel(logging.ERROR)

# General FastMCP logger can stay at WARNING
fastmcp_logger = logging.getLogger("FastMCP")
fastmcp_logger.setLevel(logging.WARNING)


@dataclass
class Server:
    """IMAS MCP Server - Composable integrator using composition pattern."""

    # Configuration parameters
    ids_set: Optional[set[str]] = None

    # Internal fields
    mcp: FastMCP = field(init=False, repr=False)
    tools: Tools = field(init=False, repr=False)
    resources: Resources = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize the MCP server after dataclass initialization."""
        self.mcp = FastMCP(name="imas")

        # Initialize components
        self.tools = Tools(ids_set=self.ids_set)
        self.resources = Resources()

        # Register components with MCP server
        self._register_components()

        logger.info("IMAS MCP Server initialized with tools and resources")

    def _register_components(self):
        """Register tools and resources with the MCP server."""
        logger.info("Registering tools component")
        self.tools.register(self.mcp)

        logger.info("Registering resources component")
        self.resources.register(self.mcp)

        logger.info("Successfully registered all components")

    def run(self, transport: str = "stdio", host: str = "127.0.0.1", port: int = 8000):
        """Run the server with the specified transport."""
        if transport == "stdio":
            logger.info("Starting IMAS MCP server with stdio transport")
            self.mcp.run()
        elif transport == "http":
            logger.info(
                f"Starting IMAS MCP server with HTTP transport on {host}:{port}"
            )
            self._run_http(host, port)
        else:
            raise ValueError(f"Unsupported transport: {transport}")

    def _run_http(self, host: str, port: int):
        """Run the server with HTTP transport."""
        try:
            import uvicorn

            # Note: HTTP transport import path may need adjustment based on FastMCP version
            from fastmcp.transports.http import create_app
        except ImportError as e:
            raise ImportError(
                "HTTP transport requires additional dependencies. "
                "Install with: pip install imas-mcp[http]"
            ) from e

        app = create_app(self.mcp)
        uvicorn.run(app, host=host, port=port, log_level="info")

    def _get_version(self) -> str:
        """Get the package version."""
        try:
            return importlib.metadata.version("imas-mcp")
        except Exception:
            return "unknown"


def main():
    """Run the server with stdio transport."""
    server = Server()
    server.run(transport="stdio")


def run_server(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8000):
    """
    Entry point for running the server with specified transport.

    Args:
        transport: Either 'stdio' or 'http'
        host: Host for HTTP transport
        port: Port for HTTP transport
    """
    server = Server()
    server.run(transport=transport, host=host, port=port)


if __name__ == "__main__":
    main()
