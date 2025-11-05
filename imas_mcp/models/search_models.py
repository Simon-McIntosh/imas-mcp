"""Search models for IMAS-Python documentation integration."""

from typing import Any

from pydantic import BaseModel, Field

from imas_mcp.models.context_models import BaseToolResult


class IMASPythonSearchMetadata(BaseModel):
    """Metadata for IMAS-Python search results."""

    version: str = Field(description="IMAS-Python version")
    source: str = Field(description="Documentation source URL")
    search_timestamp: str = Field(description="When the search was performed")
    total_results: int = Field(description="Total number of results found")


class IMASPythonSearchResult(BaseModel):
    """Single search result from IMAS-Python documentation."""

    title: str = Field(description="Title of the documentation page")
    url: str = Field(description="URL to the documentation page")
    content: str = Field(description="Content snippet from the documentation")
    relevance_score: float = Field(
        description="Relevance score (0-1)", ge=0.0, le=1.0
    )
    section: str | None = Field(
        default=None, description="Documentation section (e.g., 'API', 'Tutorial')"
    )


class IMASPythonSearchRequest(BaseModel):
    """Request model for IMAS-Python documentation search."""

    query: str = Field(
        min_length=1, max_length=500, description="Search query for IMAS-Python docs"
    )
    version: str | None = Field(
        default=None, description="Specific IMAS-Python version (auto-detected if None)"
    )
    max_results: int = Field(
        default=10, ge=1, le=50, description="Maximum number of results to return"
    )


class IMASPythonSearchResponse(BaseToolResult):
    """Response from IMAS-Python documentation search."""

    query: str = Field(description="Original search query")
    results: list[IMASPythonSearchResult] = Field(
        default_factory=list, description="Search results"
    )
    metadata: IMASPythonSearchMetadata | None = Field(
        default=None, description="Search metadata"
    )
    suggestions: list[str] = Field(
        default_factory=list, description="Search suggestions for no/few results"
    )
    error: str | None = Field(default=None, description="Error message if search failed")
    setup_instructions: str | None = Field(
        default=None, description="Setup instructions if service unavailable"
    )

    @property
    def result_count(self) -> int:
        """Number of results returned."""
        return len(self.results)

    @property
    def has_results(self) -> bool:
        """Whether any results were found."""
        return len(self.results) > 0


class DocsMCPServerStatus(BaseModel):
    """Status information for docs-mcp-server."""

    available: bool = Field(description="Whether the service is available")
    url: str = Field(description="Service URL")
    error: str | None = Field(default=None, description="Error message if unavailable")


class VersionInfo(BaseModel):
    """IMAS-Python version information."""

    version: str = Field(description="Version string")
    available: bool = Field(description="Whether this version is scraped/available")
    scraped_at: str | None = Field(
        default=None, description="When this version was scraped"
    )
    source_url: str | None = Field(
        default=None, description="Documentation source URL"
    )
