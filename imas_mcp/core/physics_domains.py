"""
Pydantic models for physics domain definitions.

This module provides Pydantic models for parsing and validating the YAML domain definitions.
Models ensure type safety and provide structured access to domain data.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator


class ComplexityLevel(str, Enum):
    """Complexity levels for physics domains."""

    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class DomainCharacteristics(BaseModel):
    """Characteristics of a physics domain."""

    description: str
    primary_phenomena: List[str]
    typical_units: List[str]
    measurement_methods: List[str]
    related_domains: List[str]
    complexity_level: ComplexityLevel

    @field_validator(
        "primary_phenomena", "typical_units", "measurement_methods", "related_domains"
    )
    @classmethod
    def validate_non_empty_lists(cls, v: List[str]) -> List[str]:
        """Ensure lists are not empty."""
        if not v:
            raise ValueError("List cannot be empty")
        return v


class PhysicsDomainMetadata(BaseModel):
    """Metadata for physics domain definitions."""

    version: str
    created: str
    description: str
    ai_assisted: bool = False
    generation_context: Optional[str] = None


class PhysicsDomainsData(BaseModel):
    """Complete physics domains data structure."""

    metadata: PhysicsDomainMetadata
    domains: Dict[str, DomainCharacteristics]

    @field_validator("domains")
    @classmethod
    def validate_domains_not_empty(
        cls, v: Dict[str, DomainCharacteristics]
    ) -> Dict[str, DomainCharacteristics]:
        """Ensure domains dictionary is not empty."""
        if not v:
            raise ValueError("Domains dictionary cannot be empty")
        return v


class IDSMappingMetadata(BaseModel):
    """Metadata for IDS mapping definitions."""

    version: str
    created: str
    description: str
    ai_assisted: bool = False
    generation_context: Optional[str] = None
    total_ids: int
    total_domains: int


class IDSMappingData(BaseModel):
    """IDS to domain mapping data structure."""

    metadata: IDSMappingMetadata
    # Dynamic fields for domain -> IDS list mappings
    model_config = {"extra": "allow"}

    def get_domain_mappings(self) -> Dict[str, List[str]]:
        """Get domain to IDS mappings, excluding metadata."""
        return {k: v for k, v in self.model_dump().items() if k != "metadata"}

    def get_ids_to_domain_mapping(self) -> Dict[str, str]:
        """Get flattened IDS to domain mapping."""
        mapping = {}
        for domain, ids_list in self.get_domain_mappings().items():
            for ids_name in ids_list:
                mapping[ids_name.lower()] = domain
        return mapping


class DomainRelationshipsMetadata(BaseModel):
    """Metadata for domain relationships definitions."""

    version: str
    created: str
    description: str
    ai_assisted: bool = False
    generation_context: Optional[str] = None


class DomainRelationshipsData(BaseModel):
    """Domain relationships data structure."""

    metadata: DomainRelationshipsMetadata
    relationships: Dict[str, List[str]]

    @field_validator("relationships")
    @classmethod
    def validate_relationships(cls, v: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Validate relationship mappings."""
        for domain, related in v.items():
            if not isinstance(related, list):
                raise ValueError(f"Related domains for {domain} must be a list")
        return v


class UnitContext(BaseModel):
    """Unit context information."""

    unit_str: str
    context: str
    category: Optional[str] = None
    physics_domains: List[str] = Field(default_factory=list)


class UnitContextsMetadata(BaseModel):
    """Metadata for unit contexts."""

    version: str
    description: str
    created_date: str
    total_units_covered: int
    source: str
    usage: str


class UnitContextsData(BaseModel):
    """Unit contexts data structure."""

    metadata: UnitContextsMetadata
    unit_contexts: Dict[str, str]
    unit_categories: Dict[str, List[str]]
    physics_domain_hints: Dict[str, List[str]]

    def get_enhanced_unit_contexts(self) -> List[UnitContext]:
        """Get enhanced unit context objects with category and domain information."""
        contexts = []

        # Create reverse mapping for unit to category
        unit_to_category = {}
        for category, units in self.unit_categories.items():
            for unit in units:
                unit_to_category[unit] = category

        for unit_str, context in self.unit_contexts.items():
            category = unit_to_category.get(unit_str)
            physics_domains = (
                self.physics_domain_hints.get(category, []) if category else []
            )

            contexts.append(
                UnitContext(
                    unit_str=unit_str,
                    context=context,
                    category=category,
                    physics_domains=physics_domains,
                )
            )

        return contexts


class PhysicsQuantity(BaseModel):
    """Enhanced physics quantity with domain information."""

    name: str
    concept: str
    description: str
    units: str
    symbol: str
    physics_domain: str
    imas_paths: List[str] = Field(default_factory=list)
    alternative_names: List[str] = Field(default_factory=list)
    coordinate_type: Optional[str] = None
    typical_ranges: Optional[Dict[str, str]] = None
    measurement_methods: List[str] = Field(default_factory=list)
    related_quantities: List[str] = Field(default_factory=list)

    @field_validator(
        "name", "concept", "description", "units", "symbol", "physics_domain"
    )
    @classmethod
    def validate_non_empty_strings(cls, v: str) -> str:
        """Ensure string fields are not empty."""
        if not v or not v.strip():
            raise ValueError("String fields cannot be empty")
        return v.strip()


class PhysicsContext(BaseModel):
    """Physics context combining all definition sources."""

    domains: PhysicsDomainsData
    ids_mapping: IDSMappingData
    relationships: DomainRelationshipsData
    unit_contexts: UnitContextsData
    physics_quantities: List[PhysicsQuantity] = Field(default_factory=list)

    def get_domain_names(self) -> Set[str]:
        """Get all available domain names."""
        return set(self.domains.domains.keys())

    def get_domain_characteristics(
        self, domain: str
    ) -> Optional[DomainCharacteristics]:
        """Get characteristics for a specific domain."""
        return self.domains.domains.get(domain)

    def get_ids_for_domain(self, domain: str) -> List[str]:
        """Get IDS names for a specific domain."""
        domain_mappings = self.ids_mapping.get_domain_mappings()
        return domain_mappings.get(domain, [])

    def get_domain_for_ids(self, ids_name: str) -> Optional[str]:
        """Get domain for a specific IDS."""
        ids_mapping = self.ids_mapping.get_ids_to_domain_mapping()
        return ids_mapping.get(ids_name.lower())

    def get_related_domains(self, domain: str) -> List[str]:
        """Get domains related to the specified domain."""
        return self.relationships.relationships.get(domain, [])

    def get_domains_by_complexity(self, complexity: ComplexityLevel) -> List[str]:
        """Get domains filtered by complexity level."""
        return [
            domain_name
            for domain_name, characteristics in self.domains.domains.items()
            if characteristics.complexity_level == complexity
        ]

    def get_domains_by_phenomenon(self, phenomenon: str) -> List[str]:
        """Get domains that include a specific phenomenon."""
        phenomenon_lower = phenomenon.lower()
        matching_domains = []

        for domain_name, characteristics in self.domains.domains.items():
            for phenom in characteristics.primary_phenomena:
                if phenomenon_lower in phenom.lower():
                    matching_domains.append(domain_name)
                    break

        return matching_domains

    def get_domains_by_unit(self, unit: str) -> List[str]:
        """Get domains that typically use a specific unit."""
        matching_domains = []

        for domain_name, characteristics in self.domains.domains.items():
            if unit in characteristics.typical_units:
                matching_domains.append(domain_name)

        return matching_domains

    def get_cross_domain_analysis(self, target_domain: str) -> Dict[str, Any]:
        """Get comprehensive cross-domain analysis for a target domain."""
        if target_domain not in self.domains.domains:
            return {"error": f"Domain '{target_domain}' not found"}

        characteristics = self.domains.domains[target_domain]
        related = self.get_related_domains(target_domain)
        ids_list = self.get_ids_for_domain(target_domain)

        # Find domains sharing phenomena
        shared_phenomena_domains = []
        for phenomenon in characteristics.primary_phenomena:
            domains = self.get_domains_by_phenomenon(phenomenon)
            shared_phenomena_domains.extend([d for d in domains if d != target_domain])

        # Find domains sharing units
        shared_units_domains = []
        for unit in characteristics.typical_units:
            domains = self.get_domains_by_unit(unit)
            shared_units_domains.extend([d for d in domains if d != target_domain])

        return {
            "target_domain": target_domain,
            "characteristics": characteristics.model_dump(),
            "direct_relationships": related,
            "ids_count": len(ids_list),
            "sample_ids": ids_list[:5],
            "shared_phenomena_domains": list(set(shared_phenomena_domains)),
            "shared_units_domains": list(set(shared_units_domains)),
            "complexity_peers": [
                d
                for d in self.get_domains_by_complexity(
                    characteristics.complexity_level
                )
                if d != target_domain
            ],
        }
