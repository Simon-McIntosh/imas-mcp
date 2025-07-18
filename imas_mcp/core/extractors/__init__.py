"""Composable extractors for IMAS data dictionary transformation."""

from .base import BaseExtractor, ExtractorContext
from .coordinate_extractor import CoordinateExtractor
from .identifier_extractor import IdentifierExtractor
from .metadata_extractor import MetadataExtractor
from .physics_extractor import LifecycleExtractor, PhysicsExtractor
from .relationship_extractor import RelationshipExtractor
from .semantic_extractor import PathExtractor, SemanticExtractor
from .validation_extractor import ValidationExtractor

__all__ = [
    "BaseExtractor",
    "ExtractorContext",
    "CoordinateExtractor",
    "IdentifierExtractor",
    "LifecycleExtractor",
    "MetadataExtractor",
    "PathExtractor",
    "PhysicsExtractor",
    "RelationshipExtractor",
    "SemanticExtractor",
    "ValidationExtractor",
]
