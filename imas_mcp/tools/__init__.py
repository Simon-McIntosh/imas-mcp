"""
IMAS MCP Tools Package.

This package contains the refactored Tools implementation split into focused modules.
Each module handles a specific tool functionality with clean separation of concerns.
"""

from typing import Optional

from fastmcp import FastMCP

from imas_mcp.providers import MCPProvider
from imas_mcp.search.document_store import DocumentStore

from .analysis_tool import AnalysisTool

# Import individual tool classes
from .base import BaseTool
from .explain_tool import ExplainTool
from .export_tool import ExportTool
from .identifiers_tool import IdentifiersTool
from .imas_python_search_tool import IMASPythonSearchTool
from .list_tool import ListTool
from .overview_tool import OverviewTool
from .path_tool import PathTool
from .relationships_tool import RelationshipsTool
from .search_tool import SearchTool


class Tools(MCPProvider):
    """Main Tools class that delegates to individual tool implementations."""

    def __init__(
        self,
        ids_set: set[str] | None = None,
        enable_imas_python_search: bool = True,
        docs_mcp_url: str = "http://localhost:3000",
        docs_db_path: str = "./docs-mcp-data",
    ):
        """Initialize the IMAS tools provider.

        Args:
            ids_set: Optional set of IDS names to limit processing to.
                    If None, will process all available IDS.
            enable_imas_python_search: Whether to enable IMAS-Python documentation search
            docs_mcp_url: URL of the docs-mcp-server
            docs_db_path: Path to documentation database
        """
        self.ids_set = ids_set
        self.enable_imas_python_search = enable_imas_python_search

        # Create shared DocumentStore with ids_set
        self.document_store = DocumentStore(ids_set=ids_set)

        # Initialize individual tools with shared document store
        self.search_tool = SearchTool(self.document_store)
        self.path_tool = PathTool(self.document_store)
        self.list_tool = ListTool(self.document_store)
        self.explain_tool = ExplainTool(self.document_store)
        self.overview_tool = OverviewTool(self.document_store)
        self.analysis_tool = AnalysisTool(self.document_store)
        self.relationships_tool = RelationshipsTool(self.document_store)
        self.identifiers_tool = IdentifiersTool(self.document_store)
        self.export_tool = ExportTool(self.document_store)

        # Initialize IMAS-Python search tool if enabled
        self.imas_python_search_tool: IMASPythonSearchTool | None = None
        if enable_imas_python_search:
            self.imas_python_search_tool = IMASPythonSearchTool(
                docs_mcp_url=docs_mcp_url,
                docs_db_path=docs_db_path,
            )

    @property
    def name(self) -> str:
        """Provider name for logging and identification."""
        return "tools"

    def register(self, mcp: FastMCP):
        """Register all IMAS tools with the MCP server."""
        # Build list of tools to register
        tools_to_register = [
            self.search_tool,
            self.path_tool,
            self.list_tool,
            self.explain_tool,
            self.overview_tool,
            self.analysis_tool,
            self.relationships_tool,
            self.identifiers_tool,
            self.export_tool,
        ]

        # Add IMAS-Python search tool if enabled
        if self.imas_python_search_tool is not None:
            tools_to_register.append(self.imas_python_search_tool)

        # Register tools from each module
        for tool in tools_to_register:
            for attr_name in dir(tool):
                attr = getattr(tool, attr_name)
                if hasattr(attr, "_mcp_tool") and attr._mcp_tool:
                    mcp.tool(description=attr._mcp_description)(attr)

    # Primary method delegation
    async def search_imas(self, *args, **kwargs):
        """Delegate to search tool."""
        return await self.search_tool.search_imas(*args, **kwargs)

    async def check_imas_paths(self, *args, **kwargs):
        """Delegate to path tool."""
        return await self.path_tool.check_imas_paths(*args, **kwargs)

    async def fetch_imas_paths(self, *args, **kwargs):
        """Delegate to path tool."""
        return await self.path_tool.fetch_imas_paths(*args, **kwargs)

    async def list_imas_paths(self, *args, **kwargs):
        """Delegate to list tool."""
        return await self.list_tool.list_imas_paths(*args, **kwargs)

    async def explain_concept(self, *args, **kwargs):
        """Delegate to explain tool."""
        return await self.explain_tool.explain_concept(*args, **kwargs)

    async def get_overview(self, *args, **kwargs):
        """Delegate to overview tool."""
        return await self.overview_tool.get_overview(*args, **kwargs)

    async def explore_identifiers(self, *args, **kwargs):
        """Delegate to identifiers tool."""
        return await self.identifiers_tool.explore_identifiers(*args, **kwargs)

    async def analyze_ids_structure(self, *args, **kwargs):
        """Delegate to analysis tool."""
        return await self.analysis_tool.analyze_ids_structure(*args, **kwargs)

    async def explore_relationships(self, *args, **kwargs):
        """Delegate to relationships tool."""
        return await self.relationships_tool.explore_relationships(*args, **kwargs)

    async def export_ids(self, *args, **kwargs):
        """Delegate to export tool."""
        return await self.export_tool.export_ids(*args, **kwargs)

    async def export_physics_domain(self, *args, **kwargs):
        """Delegate to export tool."""
        return await self.export_tool.export_physics_domain(*args, **kwargs)

    async def search_imas_python_docs(self, *args, **kwargs):
        """Delegate to IMAS-Python search tool."""
        if self.imas_python_search_tool is None:
            raise RuntimeError("IMAS-Python search tool is not enabled")
        return await self.imas_python_search_tool.search_imas_python_docs(*args, **kwargs)


__all__ = [
    "BaseTool",
    "SearchTool",
    "PathTool",
    "ListTool",
    "ExplainTool",
    "OverviewTool",
    "AnalysisTool",
    "RelationshipsTool",
    "IdentifiersTool",
    "ExportTool",
    "IMASPythonSearchTool",
    "Tools",
]
