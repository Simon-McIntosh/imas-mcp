#!/usr/bin/env python3
"""
Build the schema data structures for the IMAS Data Dictionary.
This script transforms the XML data dictionary into JSON format and saves it to resources.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import click

from imas_mcp.core.xml_parser import DataDictionaryTransformer


@click.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress all logging except errors")
@click.option(
    "--force", "-f", is_flag=True, help="Force rebuild even if files already exist"
)
@click.option(
    "--ids-filter",
    type=str,
    help="Specific IDS names to include as a space-separated string (e.g., 'core_profiles equilibrium')",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    help="Custom output directory (defaults to imas_mcp/resources/schemas)",
)
@click.option(
    "--no-rich",
    is_flag=True,
    help="Disable rich progress display and use plain logging",
)
def build_schemas(
    verbose: bool,
    quiet: bool,
    force: bool,
    ids_filter: str,
    output_dir: Optional[Path],
    no_rich: bool,
) -> int:
    """Build the schema data structures for the IMAS Data Dictionary.

    This command initializes a DataDictionaryTransformer and generates schema files
    containing structured data dictionary information optimized for LLM processing.

    Examples:
        build-schemas                    # Build schema data with default settings
        build-schemas -v                 # Build with verbose logging
        build-schemas -f                 # Force rebuild even if exists
        build-schemas --ids-filter "core_profiles equilibrium"  # Build specific IDS only
        build-schemas --output-dir /path/to/custom/dir  # Use custom output directory
        build-schemas --no-rich          # Disable rich progress and use plain logging
    """
    # Set up logging level
    if quiet:
        log_level = logging.ERROR
    elif verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting schema data build process...")

        # Parse ids_filter string into a set if provided
        ids_set: Optional[set] = set(ids_filter.split()) if ids_filter else None
        if ids_set:
            logger.info(f"Building schema data for specific IDS: {sorted(ids_set)}")
        else:
            logger.info("Building schema data for all available IDS")

        # Initialize the transformer
        transformer = DataDictionaryTransformer(
            output_dir=output_dir,
            ids_set=ids_set,
            use_rich=not no_rich,  # Invert no_rich flag
        )

        # Check if we need to build
        catalog_file = transformer.resolved_output_dir / "ids_catalog.json"
        should_build = force or not catalog_file.exists()

        if should_build:
            if force and catalog_file.exists():
                logger.info("Force rebuilding existing schema data files")
            else:
                logger.info("Schema data files do not exist, building new files...")

            # Transform the data
            logger.info("Starting XML to JSON transformation...")
            outputs = transformer.transform_complete()

            # Log the results
            logger.info("Schema data built successfully:")
            logger.info(f"  - Catalog: {outputs.catalog}")
            logger.info(f"  - Relationships: {outputs.relationships}")
            logger.info(f"  - Identifier Catalog: {outputs.identifier_catalog}")
            logger.info(f"  - Detailed files: {len(outputs.detailed)} files")

            # Print summary for scripts/CI
            total_files = 3 + len(
                outputs.detailed
            )  # catalog + relationships + identifier_catalog + detailed files
            click.echo(
                f"Built {total_files} schema files in {transformer.resolved_output_dir}"
            )
        else:
            logger.info(
                f"Schema data files already exist in {transformer.resolved_output_dir}"
            )
            click.echo(
                f"Schema data already exists in {transformer.resolved_output_dir}"
            )

        return 0

    except Exception as e:
        logger.error(f"Error building schema data: {e}")
        if verbose:
            logger.exception("Full traceback:")
        click.echo(f"Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(build_schemas())
