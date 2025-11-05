"""Extended error models for IMAS-Python integration."""

from typing import Any


class DocsMCPServerUnavailableError(Exception):
    """Error raised when docs-mcp-server is unavailable."""

    def __init__(
        self,
        message: str,
        setup_instructions: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.setup_instructions = setup_instructions
        self.context = context or {}


class VersionNotAvailableError(Exception):
    """Error raised when requested IMAS-Python version is not available."""

    def __init__(
        self,
        version: str,
        available_versions: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ):
        self.version = version
        self.available_versions = available_versions or []
        self.context = context or {}
        message = f"IMAS-Python version '{version}' not available"
        if available_versions:
            message += f". Available versions: {', '.join(available_versions[:5])}"
        super().__init__(message)


class ScrapingFailedError(Exception):
    """Error raised when documentation scraping fails."""

    def __init__(
        self, message: str, version: str | None = None, context: dict[str, Any] | None = None
    ):
        super().__init__(message)
        self.version = version
        self.context = context or {}
