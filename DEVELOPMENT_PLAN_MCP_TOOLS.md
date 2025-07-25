# IMAS MCP Tools Development Plan

## Executive Summary

This plan implements critical improvements to the IMAS MCP server based on comprehensive tool analysis, focusing on performance optimization, selective AI enhancement, and expanded capabilities with MCP resources and prompts.

## Phase 0: Testing Foundation & Performance Baseline (Weeks 1-2)

### 0.1 Pytest Unit Tests for Current MCP Tools

**Purpose**: Establish comprehensive test coverage for existing tools before refactoring

**Status**: ✅ **IMPLEMENTED**

**Test Infrastructure**: Implemented in `tests/conftest.py` with:

- `TestServer` class with cached properties for performance
- Session-scoped fixtures for expensive operations
- Integration with FastMCP Client for realistic testing
- Sample data fixtures for consistent test data

**Comprehensive Test Suite**: Implemented in `tests/test_server_tools.py` with:

- `TestSearchImas` - Complete search functionality tests
- `TestGetOverview` - Overview tool tests
- `TestAnalyzeIDSStructure` - IDS structure analysis tests
- `TestExploreRelationships` - Relationship exploration tests
- `TestExploreIdentifiers` - Identifier exploration tests
- `TestExportIDS` - Bulk export tests
- `TestExportPhysicsDomain` - Physics domain export tests
- `TestExplainConcept` - Concept explanation tests
- `TestServerErrorHandling` - Error handling tests

**Test Coverage**: 629 lines of comprehensive test coverage for all current MCP tools with both fast and integration test markers.

**Files implemented**: `tests/conftest.py`, `tests/test_server_tools.py`
**Status**: ✅ **IMPLEMENTED** - All test classes implemented in `tests/test_server_tools.py`:

- TestSearchImas: Tests for search functionality with various query types and filters
- TestExplainConcept: Tests for concept explanation with different detail levels
- TestGetOverview: Tests for overview functionality with and without questions
- TestAnalyzeIDSStructure: Tests for IDS structure analysis with valid/invalid IDS
- TestExploreRelationships: Tests for relationship exploration with different depths
- TestExploreIdentifiers: Tests for identifier exploration with queries
- TestExportIDS: Tests for bulk export functionality with relationships
- TestExportPhysicsDomain: Tests for physics domain export with cross-domain analysis
- TestIntegration: Integration tests for tool workflows (search->explain, search->structure, bulk export)

### 0.2 ASV Performance Monitoring Setup

**Purpose**: Establish performance baselines and regression detection

**Status**: ✅ **IMPLEMENTED** - ASV configured with uv and performance monitoring ready

**Files implemented**:

- `benchmarks/asv.conf.json` - ASV configuration with uv integration
- `benchmarks/benchmarks.py` - Comprehensive benchmark suite
- `benchmarks/benchmark_runner.py` - Utility for running ASV benchmarks
- `benchmarks/performance_targets.py` - Performance targets and validation
- `scripts/run_performance_baseline.py` - Baseline establishment script
- Updated `Makefile` with benchmark targets

### 0.3 Test and Performance Integration

**Status**: ✅ COMPLETE  
**Objective**: Integrate automated testing and performance monitoring into development workflow

**Context**: Essential CI/CD pipeline for maintaining quality and performance standards across development iterations.

**Implementation Summary**:

- ✅ **Benchmark Dependencies**: Added `bench = ["asv[virtualenv]>=0.6.0,<1.0.0"]` to pyproject.toml optional dependencies
- ✅ **Makefile Targets**: Created install-bench, performance-baseline, performance-current, and performance-compare targets
- ✅ **GitHub Actions Workflow**: Implemented comprehensive `.github/workflows/benchmarks.yml` with:
  - Automated benchmarks on push/PR to main/develop branches
  - Daily scheduled runs at 2 AM UTC
  - Manual trigger with benchmark filtering capability
  - ASV result artifacts with 90-day retention
  - GitHub Pages deployment for benchmark reports
  - PR comments with benchmark result links
  - Full history fetch for ASV comparison features

**Key Features Delivered**:

- Cross-platform benchmark execution with UV package manager
- Automatic GitHub Pages deployment to `https://{owner}.github.io/{repo}/benchmarks/`
- Performance comparison between commits and branches
- Persistent benchmark history with artifact storage
- Interactive HTML dashboard with performance trends
- PR integration with automated benchmark result posting

**GitHub Actions Status**: Active workflows include benchmarks.yml, docker-build-push.yml, release.yml, and test.yml providing comprehensive CI/CD coverage.

### 0.4 Performance Targets and Monitoring

**Purpose**: Define performance targets and monitoring strategy

**Status**: ✅ **IMPLEMENTED** - Replaced by ASV performance monitoring system

**ASV Integration**: Comprehensive performance monitoring implemented with:

- **ASV Configuration**: Complete setup in `asv.conf.json` with Python 3.12 environment
- **Benchmark Suite**: Implemented in `benchmarks/benchmarks.py` with comprehensive test classes:
  - `SearchBenchmarks` - Search performance tests (7-10ms timing)
  - `ExplainConceptBenchmarks` - Concept explanation performance
  - `StructureAnalysisBenchmarks` - IDS structure analysis performance
  - `BulkExportBenchmarks` - Bulk export performance tests
  - `RelationshipBenchmarks` - Relationship exploration performance
- **Performance Baseline Script**: Automated script in `scripts/run_performance_baseline.py` with:
  - Click-based CLI with filtering options
  - ASV machine configuration and benchmark execution
  - HTML report generation and serving instructions
  - CI-aware progress display
- **Benchmark Runner**: Utility class in `benchmarks/benchmark_runner.py` for programmatic ASV execution
- **HTML Dashboard**: Interactive performance results at `.asv/html/index.html`

**Key files implemented**:

- `asv.conf.json` - ASV configuration with cross-platform path handling
- `benchmarks/benchmarks.py` - Comprehensive benchmark definitions
- `benchmarks/benchmark_runner.py` - ASV wrapper utility
- `scripts/run_performance_baseline.py` - Automated benchmark execution script

**Performance Monitoring Features**:

- Automatic benchmark discovery and execution
- Performance regression detection
- Interactive HTML dashboard with graphs and trends
- Memory usage monitoring (peakmem benchmarks)
- CI/CD integration support

**Replaced Features**: The original `benchmarks/performance_targets.py` file with manual performance targets was removed in favor of ASV's superior automatic performance tracking and regression detection capabilities.

## Phase 1: Core Tool Optimization (Weeks 3-5)

### 1.1 Enhanced search_imas Tool (Priority 1)

**Status**: ✅ **IMPLEMENTED** - Search performance optimization with caching and search modes

**Search Modes**: Implemented with `SearchMode` enum:

- `AUTO` - Matches planned "adaptive" mode (intelligent search selection)
- `SEMANTIC` - Matches planned "comprehensive" mode (full AI-powered search)
- `LEXICAL` - Matches planned "fast" mode (traditional text search)
- `HYBRID` - Advanced combination mode not in original plan

**Caching System**: Fully implemented using `cachetools`:

- `SearchCache` class with TTL and size limits (1000 items, 1 hour TTL)
- Integrated into `search_imas` method with cache check/set
- Performance statistics tracking (hits, misses, sets, hit rate)
- Comprehensive test coverage in `tests/test_search_cache.py` and `tests/test_search_cache_integration.py`

**Key files implemented**:

- `imas_mcp/search/cache.py` - SearchCache class using cachetools
- `imas_mcp/search/search_modes.py` - SearchMode enum and search strategies
- `imas_mcp/server.py` - Integrated caching into search_imas method
- `tests/test_search_cache.py` - Unit tests for caching functionality
- `tests/test_search_cache_integration.py` - Integration tests with server

**Main functionality covered**:

- Performance-optimized search with automatic caching
- Search strategies (semantic, lexical, hybrid, auto)
- Cache statistics and monitoring for performance analysis
- TTL-based cache timeout and size-based eviction

**Remaining unimplemented features**:

```python
def _suggest_follow_up_tools(self, results: Dict[str, Any]) -> List[Dict[str, str]]:
    """Suggest relevant tools based on current results."""
    suggestions = []

    # Physics concept suggestions
    if results.get("physics_matches"):
        suggestions.append({
            "tool": "explain_concept",
            "reason": "Get detailed physics explanations for found concepts",
            "sample_call": f"explain_concept('{results['physics_matches'][0]['concept']}')"
        })

    # Relationship analysis suggestions
    if len(results.get("results", [])) > 1:
        unique_ids = set(r["ids_name"] for r in results["results"])
        if len(unique_ids) > 1:
            suggestions.append({
                "tool": "explore_relationships",
                "reason": "Analyze relationships between found paths across IDS",
                "sample_call": f"explore_relationships('{results['results'][0]['path']}')"
            })

    # Bulk export suggestions
    if len(results.get("results", [])) > 5:
        unique_ids = list(set(r["ids_name"] for r in results["results"]))
        if len(unique_ids) > 1:
            suggestions.append({
                "tool": "export_ids",
                "reason": f"Export bulk data for {len(unique_ids)} IDS with relationships",
                "sample_call": f"export_ids({unique_ids[:3]})"
            })

    return suggestions

# Parameter needed in search_imas:
# include_suggestions: bool = True,
```

### 1.2 Selective AI Enhancement Strategy

**Status**: ✅ **IMPLEMENTED**

**AI Enhancement Strategy**: Comprehensive selective AI enhancement system implemented with conditional logic for optimal performance.

**Key files implemented**:

- `imas_mcp/search/ai_enhancer.py` - AI enhancement decorator with selective strategy
- `imas_mcp/search/ai_enhancement_strategy.py` - Conditional enhancement logic and tool suggestions
- `tests/test_enhancement_strategy.py` - Unit tests for enhancement strategy logic
- `tests/test_enhancer_integration.py` - Integration tests with server tools

**Main functionality covered**:

- **AI Enhancement Strategy Mapping**: All 8 tools configured with appropriate strategies:
  - `search_imas`, `analyze_ids_structure`, `explore_relationships`, `export_ids`, `export_physics_domain`: "conditional"
  - `explain_concept`, `get_overview`: "always"
  - `explore_identifiers`: "never"
- **Conditional Enhancement Logic**: Smart evaluation based on search modes, export formats, analysis depth, and complexity
- **Multi-Format Export Support**: `export_ids` supports raw, structured, and enhanced formats with AI only for enhanced format
- **Tool Suggestions**: Automatic follow-up tool recommendations for all tools
- **Graceful Degradation**: Handles AI service unavailability without blocking tool functionality
- **Comprehensive Test Coverage**: 629 lines of tests covering all conditional logic scenarios

### 1.3 Multi-Format Export Tools

**Status**: ✅ **IMPLEMENTED**

**Multi-Format Export System**: Comprehensive format-based processing implemented in `export_ids` method.

**Key files implemented**:

- `imas_mcp/server.py` - Multi-format export logic integrated into `export_ids` method
- `tests/test_enhancer_integration.py` - Integration tests for all three export formats

**Main functionality covered**:

- **Format Validation**: Validates export format parameter with clear error messages and suggestions
- **Raw Format**: Minimal processing with disabled relationships and physics context for maximum performance
- **Structured Format**: Organized data with relationships and physics context but no AI enhancement
- **Enhanced Format**: AI-enhanced insights with comprehensive data analysis (conditional on context availability)
- **Format-Specific Processing**: Different levels of data detail based on format (truncated documentation for non-comprehensive formats)
- **Graceful Fallback**: Enhanced format falls back to structured when AI context unavailable
- **Performance Optimization**: Raw format optimized for speed with minimal data processing

## Phase 2: MCP Resources Implementation (Weeks 6-8)

### 2.1 Static JSON IDS Data Resources

**Purpose**: Serve pre-computed IDS data for efficient access

```python
# File: imas_mcp/resources/ids_catalog.py
from fastmcp import FastMCP
from typing import Dict, Any, Optional
import json
from pathlib import Path

class IDSCatalogResource:
    """MCP resource for serving static IDS catalog data."""

    def __init__(self, server: 'Server'):
        self.server = server
        self.catalog_path = Path("imas_mcp/resources/data/ids_catalog.json")
        self.structure_path = Path("imas_mcp/resources/data/ids_structures/")
        self._ensure_catalog_exists()

    def register_resources(self, mcp: FastMCP):
        """Register IDS catalog resources with MCP server."""

        @mcp.resource("ids://catalog")
        async def get_ids_catalog() -> str:
            """Get complete IDS catalog with metadata."""
            return self._load_catalog()

        @mcp.resource("ids://structure/{ids_name}")
        async def get_ids_structure(ids_name: str) -> str:
            """Get detailed structure for specific IDS."""
            return self._load_ids_structure(ids_name)

        @mcp.resource("ids://summary")
        async def get_ids_summary() -> str:
            """Get high-level summary of all IDS."""
            return self._generate_summary()

        @mcp.resource("ids://physics-domains")
        async def get_physics_domains() -> str:
            """Get physics domain mappings."""
            return self._load_physics_domains()

    def _load_catalog(self) -> str:
        """Load pre-computed IDS catalog."""
        if not self.catalog_path.exists():
            self._build_catalog()

        with open(self.catalog_path, 'r') as f:
            return f.read()

    def _build_catalog(self) -> None:
        """Build comprehensive IDS catalog."""
        available_ids = self.server.document_store.get_available_ids()

        catalog = {
            "metadata": {
                "generated": datetime.now().isoformat(),
                "total_ids": len(available_ids),
                "version": "1.0"
            },
            "ids": {}
        }

        for ids_name in available_ids:
            ids_docs = self.server.document_store.get_documents_by_ids(ids_name)

            # Calculate statistics
            physics_domains = set()
            units_distribution = {}
            identifier_count = 0

            for doc in ids_docs:
                if doc.metadata.physics_domain:
                    physics_domains.add(doc.metadata.physics_domain)
                if doc.metadata.units:
                    units_distribution[doc.metadata.units] = units_distribution.get(doc.metadata.units, 0) + 1
                if doc.raw_data.get("identifier_schema"):
                    identifier_count += 1

            catalog["ids"][ids_name] = {
                "path_count": len(ids_docs),
                "physics_domains": list(physics_domains),
                "units_distribution": dict(sorted(units_distribution.items(), key=lambda x: x[1], reverse=True)[:10]),
                "identifier_paths": identifier_count,
                "max_depth": max(len(doc.metadata.path_name.split("/")) for doc in ids_docs) if ids_docs else 0,
                "sample_paths": [doc.metadata.path_name for doc in ids_docs[:5]]
            }

        # Save catalog
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.catalog_path, 'w') as f:
            json.dump(catalog, f, indent=2)

# File: imas_mcp/resources/measurement_data.py
class MeasurementDataResource:
    """MCP resource for physics measurement data."""

    def register_resources(self, mcp: FastMCP):
        """Register measurement data resources."""

        @mcp.resource("measurement://units/{unit_type}")
        async def get_unit_paths(unit_type: str) -> str:
            """Get all paths using specific unit type."""
            return self._get_paths_by_unit(unit_type)

        @mcp.resource("measurement://coordinates/{coordinate}")
        async def get_coordinate_paths(coordinate: str) -> str:
            """Get all paths using specific coordinate."""
            return self._get_paths_by_coordinate(coordinate)

        @mcp.resource("measurement://physics-domain/{domain}")
        async def get_domain_paths(domain: str) -> str:
            """Get all paths in specific physics domain."""
            return self._get_paths_by_domain(domain)
```

### 2.2 Usage Examples Resource

**Purpose**: Provide code examples and workflows

```python
# File: imas_mcp/resources/examples.py
class ExamplesResource:
    """MCP resource for usage examples and workflows."""

    def register_resources(self, mcp: FastMCP):
        """Register example resources."""

        @mcp.resource("examples://search-patterns")
        async def get_search_patterns() -> str:
            """Get common search patterns and examples."""
            return json.dumps({
                "basic_search": {
                    "example": "search_imas('plasma temperature')",
                    "description": "Simple concept search",
                    "expected_results": "Temperature-related paths across IDS"
                },
                "field_specific": {
                    "example": "search_imas('units:eV AND documentation:electron')",
                    "description": "Field-specific search with boolean operators",
                    "expected_results": "Electron-related paths with eV units"
                },
                "bulk_search": {
                    "example": "search_imas(['plasma', 'temperature', 'density'])",
                    "description": "Multi-term search",
                    "expected_results": "Paths related to all terms"
                }
            })

        @mcp.resource("examples://workflows/{workflow_type}")
        async def get_workflow_examples(workflow_type: str) -> str:
            """Get workflow examples for specific analysis types."""
            workflows = {
                "equilibrium_analysis": {
                    "steps": [
                        "search_imas('equilibrium profiles')",
                        "analyze_ids_structure('equilibrium')",
                        "explore_relationships('equilibrium/time_slice/profiles_1d')",
                        "export_physics_domain('equilibrium')"
                    ],
                    "description": "Complete equilibrium analysis workflow"
                },
                "transport_study": {
                    "steps": [
                        "search_imas('transport coefficients')",
                        "export_ids(['core_profiles', 'transport', 'core_transport'])",
                        "explore_relationships('transport/model/diffusion')"
                    ],
                    "description": "Transport physics study workflow"
                }
            }
            return json.dumps(workflows.get(workflow_type, {}))
```

## Phase 3: MCP Prompts Implementation (Weeks 9-10)

### 3.1 Physics Analysis Prompts

**Purpose**: Provide specialized prompts for physics analysis

```python
# File: imas_mcp/prompts/physics_analysis.py
class PhysicsAnalysisPrompts:
    """MCP prompts for physics analysis tasks."""

    def register_prompts(self, mcp: FastMCP):
        """Register physics analysis prompts."""

        @mcp.prompt("physics-explain")
        async def physics_explain_prompt(
            concept: str,
            context: str = "",
            level: str = "intermediate"
        ) -> str:
            """Generate physics explanation prompt."""
            return f"""
            Explain the plasma physics concept '{concept}' at {level} level.

            Context: {context}

            Provide:
            1. Physical definition and significance
            2. Mathematical formulation (if applicable)
            3. Typical measurement methods
            4. Relationship to other plasma phenomena
            5. Practical applications in fusion research

            Format as clear, educational explanation suitable for plasma physicists.
            """

        @mcp.prompt("measurement-workflow")
        async def measurement_workflow_prompt(
            measurement_type: str,
            ids_involved: str,
            analysis_goal: str
        ) -> str:
            """Generate measurement workflow analysis prompt."""
            return f"""
            Design a measurement workflow for {measurement_type} analysis.

            Available IDS: {ids_involved}
            Analysis Goal: {analysis_goal}

            Provide:
            1. Data collection strategy
            2. Required coordinate systems
            3. Preprocessing steps
            4. Analysis methodology
            5. Validation approaches
            6. Expected outputs and interpretation

            Focus on practical implementation using IMAS data structures.
            """

        @mcp.prompt("cross-ids-analysis")
        async def cross_ids_analysis_prompt(
            ids_list: str,
            research_question: str
        ) -> str:
            """Generate cross-IDS analysis prompt."""
            return f"""
            Analyze relationships between IDS: {ids_list}
            Research Question: {research_question}

            Provide:
            1. Data coupling mechanisms
            2. Coordinate transformation requirements
            3. Consistency checks needed
            4. Integration challenges
            5. Recommended analysis sequence
            6. Quality assurance procedures

            Focus on physics-consistent analysis across multiple IDS.
            """

# File: imas_mcp/prompts/code_generation.py
class CodeGenerationPrompts:
    """MCP prompts for code generation tasks."""

    def register_prompts(self, mcp: FastMCP):
        """Register code generation prompts."""

        @mcp.prompt("imas-python-code")
        async def imas_python_code_prompt(
            task_description: str,
            paths: str,
            analysis_type: str = "basic"
        ) -> str:
            """Generate Python code for IMAS data analysis."""
            return f"""
            Generate Python code for IMAS data analysis.

            Task: {task_description}
            IMAS Paths: {paths}
            Analysis Type: {analysis_type}

            Requirements:
            1. Use IMAS Python API appropriately
            2. Include proper error handling
            3. Add physics-aware data validation
            4. Include plotting/visualization if relevant
            5. Add documentation and comments

            Generate clean, production-ready Python code.
            """

        @mcp.prompt("data-validation")
        async def data_validation_prompt(
            data_description: str,
            physics_constraints: str
        ) -> str:
            """Generate data validation approach."""
            return f"""
            Design data validation for: {data_description}
            Physics Constraints: {physics_constraints}

            Provide:
            1. Range checks and physical limits
            2. Consistency checks across related quantities
            3. Unit conversion validation
            4. Coordinate system checks
            5. Temporal consistency validation
            6. Error detection strategies

            Focus on physics-based validation rules.
            """
```

### 3.2 Workflow Automation Prompts

**Purpose**: Automate complex analysis workflows

```python
# File: imas_mcp/prompts/workflow_automation.py
class WorkflowAutomationPrompts:
    """MCP prompts for workflow automation."""

    def register_prompts(self, mcp: FastMCP):
        """Register workflow automation prompts."""

        @mcp.prompt("analysis-pipeline")
        async def analysis_pipeline_prompt(
            objective: str,
            available_data: str,
            constraints: str = ""
        ) -> str:
            """Generate complete analysis pipeline."""
            return f"""
            Design analysis pipeline for: {objective}
            Available Data: {available_data}
            Constraints: {constraints}

            Generate:
            1. Data extraction strategy
            2. Preprocessing pipeline
            3. Analysis methodology
            4. Quality control steps
            5. Visualization approach
            6. Results interpretation
            7. Error propagation

            Provide step-by-step implementation plan.
            """

        @mcp.prompt("experiment-design")
        async def experiment_design_prompt(
            research_question: str,
            hypothesis: str,
            available_diagnostics: str
        ) -> str:
            """Generate experiment design recommendations."""
            return f"""
            Design experiment for research question: {research_question}
            Hypothesis: {hypothesis}
            Available Diagnostics: {available_diagnostics}

            Provide:
            1. Measurement requirements
            2. Temporal resolution needs
            3. Spatial coverage requirements
            4. Calibration procedures
            5. Data acquisition strategy
            6. Analysis plan
            7. Expected outcomes

            Focus on physics-driven experimental design.
            """
```

## Phase 4: Performance Optimization (Weeks 11-12)

### 4.1 Caching Strategy

**Purpose**: Implement intelligent caching for common operations

```python
# File: imas_mcp/performance/cache_manager.py
from typing import Dict, Any, Optional
import hashlib
import pickle
from pathlib import Path
import time

class CacheManager:
    """Intelligent caching for MCP operations."""

    def __init__(self, cache_dir: Path = Path("cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache: Dict[str, Any] = {}
        self.cache_stats = {"hits": 0, "misses": 0}

    def get(self, key: str, ttl: int = 3600) -> Optional[Any]:
        """Get cached item with TTL check."""
        # Check memory cache first
        if key in self.memory_cache:
            item = self.memory_cache[key]
            if time.time() - item["timestamp"] < ttl:
                self.cache_stats["hits"] += 1
                return item["data"]
            else:
                del self.memory_cache[key]

        # Check disk cache
        cache_file = self.cache_dir / f"{self._hash_key(key)}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    item = pickle.load(f)
                    if time.time() - item["timestamp"] < ttl:
                        self.memory_cache[key] = item
                        self.cache_stats["hits"] += 1
                        return item["data"]
                    else:
                        cache_file.unlink()
            except Exception:
                pass

        self.cache_stats["misses"] += 1
        return None

    def set(self, key: str, data: Any, ttl: int = 3600) -> None:
        """Set cached item."""
        item = {"data": data, "timestamp": time.time()}

        # Store in memory cache
        self.memory_cache[key] = item

        # Store in disk cache for persistence
        cache_file = self.cache_dir / f"{self._hash_key(key)}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(item, f)
        except Exception:
            pass

    def _hash_key(self, key: str) -> str:
        """Generate hash for cache key."""
        return hashlib.md5(key.encode()).hexdigest()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total if total > 0 else 0

        return {
            "hit_rate": hit_rate,
            "total_requests": total,
            "memory_items": len(self.memory_cache),
            "disk_items": len(list(self.cache_dir.glob("*.pkl")))
        }
```

### 4.2 Performance Monitoring

**Purpose**: Track and optimize tool performance

```python
# File: imas_mcp/performance/monitor.py
from dataclasses import dataclass
from typing import Dict, List, Any
import time
import functools
from collections import defaultdict

@dataclass
class PerformanceMetrics:
    """Performance metrics for MCP tools."""
    tool_name: str
    execution_time: float
    cache_hit: bool
    result_count: int
    error_occurred: bool
    timestamp: float

class PerformanceMonitor:
    """Monitor and analyze MCP tool performance."""

    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.tool_stats = defaultdict(list)

    def monitor_tool(self, tool_name: str):
        """Decorator to monitor tool performance."""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                error_occurred = False
                result_count = 0
                cache_hit = False

                try:
                    result = await func(*args, **kwargs)

                    # Extract metrics from result
                    if isinstance(result, dict):
                        result_count = len(result.get("results", []))
                        cache_hit = result.get("cache_hit", False)

                    return result

                except Exception as e:
                    error_occurred = True
                    raise e

                finally:
                    execution_time = time.time() - start_time

                    # Record metrics
                    metric = PerformanceMetrics(
                        tool_name=tool_name,
                        execution_time=execution_time,
                        cache_hit=cache_hit,
                        result_count=result_count,
                        error_occurred=error_occurred,
                        timestamp=time.time()
                    )

                    self.metrics.append(metric)
                    self.tool_stats[tool_name].append(metric)

            return wrapper
        return decorator

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "overall_stats": {
                "total_requests": len(self.metrics),
                "average_response_time": sum(m.execution_time for m in self.metrics) / len(self.metrics) if self.metrics else 0,
                "error_rate": sum(1 for m in self.metrics if m.error_occurred) / len(self.metrics) if self.metrics else 0,
                "cache_hit_rate": sum(1 for m in self.metrics if m.cache_hit) / len(self.metrics) if self.metrics else 0
            },
            "tool_breakdown": {}
        }

        for tool_name, metrics in self.tool_stats.items():
            report["tool_breakdown"][tool_name] = {
                "request_count": len(metrics),
                "avg_response_time": sum(m.execution_time for m in metrics) / len(metrics),
                "error_rate": sum(1 for m in metrics if m.error_occurred) / len(metrics),
                "cache_hit_rate": sum(1 for m in metrics if m.cache_hit) / len(metrics),
                "avg_result_count": sum(m.result_count for m in metrics) / len(metrics)
            }

        return report
```

## Phase 5: Testing Strategy (Weeks 13-14)

### 5.1 Tool Testing with FastMCP 2

**Purpose**: Comprehensive testing of MCP tools

```python
# File: tests/test_mcp_tools.py
import pytest
from fastmcp import FastMCP
from fastmcp.testing import MockContext
from imas_mcp.server import Server
import json

class TestMCPTools:
    """Test suite for MCP tools using FastMCP 2."""

    @pytest.fixture
    def server(self):
        """Create test server instance."""
        return Server()

    @pytest.fixture
    def mock_context(self):
        """Create mock MCP context."""
        return MockContext()

    @pytest.mark.asyncio
    async def test_search_imas_fast_mode(self, server, mock_context):
        """Test search_imas in fast mode."""
        result = await server.search_imas(
            query="plasma temperature",
            search_mode="fast",
            max_results=5,
            ctx=mock_context
        )

        assert result["search_strategy"] == "lexical_search"
        assert result["performance_mode"] == "fast"
        assert len(result["results"]) <= 5
        assert "suggested_tools" in result

    @pytest.mark.asyncio
    async def test_search_imas_adaptive_mode(self, server, mock_context):
        """Test search_imas in adaptive mode."""
        result = await server.search_imas(
            query="equilibrium profiles",
            search_mode="adaptive",
            max_results=10,
            ctx=mock_context
        )

        assert result["search_strategy"] in ["lexical_search", "semantic_search"]
        assert len(result["results"]) <= 10
        assert "suggested_tools" in result

    @pytest.mark.asyncio
    async def test_search_imas_comprehensive_mode(self, server, mock_context):
        """Test search_imas in comprehensive mode."""
        result = await server.search_imas(
            query="transport coefficients",
            search_mode="comprehensive",
            max_results=15,
            ctx=mock_context
        )

        assert result["search_strategy"] == "semantic_search"
        assert len(result["results"]) <= 15
        assert "ai_enhancement" in result
        assert "suggested_tools" in result

    @pytest.mark.asyncio
    async def test_export_ids_raw_format(self, server):
        """Test bulk export in raw format."""
        result = await server.export_ids(
            ids_list=["core_profiles", "equilibrium"],
            format="raw",
            include_relationships=True
        )

        assert "ai_enhancement" not in result
        assert result["export_format"] == "raw"
        assert len(result["valid_ids"]) == 2

    @pytest.mark.asyncio
    async def test_export_ids_enhanced_format(self, server, mock_context):
        """Test bulk export in enhanced format."""
        result = await server.export_ids(
            ids_list=["core_profiles"],
            format="enhanced",
            include_relationships=True,
            ctx=mock_context
        )

        assert result["export_format"] == "enhanced"
        assert "ai_enhancement" in result

    @pytest.mark.asyncio
    async def test_tool_suggestions(self, server, mock_context):
        """Test tool suggestion functionality."""
        # Search for physics concepts
        search_result = await server.search_imas(
            query="plasma temperature density",
            search_mode="adaptive",
            ctx=mock_context
        )

        suggestions = search_result["suggested_tools"]
        assert len(suggestions) > 0

        # Check suggestion structure
        for suggestion in suggestions:
            assert "tool" in suggestion
            assert "reason" in suggestion
            assert "sample_call" in suggestion

    @pytest.mark.asyncio
    async def test_conditional_ai_enhancement(self, server, mock_context):
        """Test conditional AI enhancement strategy."""
        # Test tool that should not have AI enhancement
        result = await server.explore_identifiers(
            query="temperature",
            scope="summary",
            ctx=mock_context
        )

        assert "ai_enhancement" not in result

        # Test tool that should have AI enhancement
        result = await server.explain_concept(
            concept="plasma temperature",
            ctx=mock_context
        )

        assert "ai_enhancement" in result

# File: tests/test_mcp_resources.py
import pytest
from fastmcp.testing import MockResourceClient
from imas_mcp.resources.ids_catalog import IDSCatalogResource
import json

class TestMCPResources:
    """Test suite for MCP resources."""

    @pytest.fixture
    def resource_client(self):
        """Create mock resource client."""
        return MockResourceClient()

    @pytest.mark.asyncio
    async def test_ids_catalog_resource(self, resource_client):
        """Test IDS catalog resource."""
        response = await resource_client.get("ids://catalog")

        catalog = json.loads(response)
        assert "metadata" in catalog
        assert "ids" in catalog
        assert catalog["metadata"]["total_ids"] > 0

    @pytest.mark.asyncio
    async def test_ids_structure_resource(self, resource_client):
        """Test IDS structure resource."""
        response = await resource_client.get("ids://structure/core_profiles")

        structure = json.loads(response)
        assert "path_count" in structure
        assert "physics_domains" in structure
        assert "sample_paths" in structure

    @pytest.mark.asyncio
    async def test_physics_domains_resource(self, resource_client):
        """Test physics domains resource."""
        response = await resource_client.get("ids://physics-domains")

        domains = json.loads(response)
        assert isinstance(domains, dict)
        assert len(domains) > 0

# File: tests/test_mcp_prompts.py
import pytest
from fastmcp.testing import MockPromptClient
from imas_mcp.prompts.physics_analysis import PhysicsAnalysisPrompts

class TestMCPPrompts:
    """Test suite for MCP prompts."""

    @pytest.fixture
    def prompt_client(self):
        """Create mock prompt client."""
        return MockPromptClient()

    @pytest.mark.asyncio
    async def test_physics_explain_prompt(self, prompt_client):
        """Test physics explanation prompt."""
        response = await prompt_client.get_prompt(
            "physics-explain",
            concept="plasma temperature",
            level="intermediate"
        )

        assert "plasma temperature" in response
        assert "intermediate" in response
        assert "Physical definition" in response

    @pytest.mark.asyncio
    async def test_measurement_workflow_prompt(self, prompt_client):
        """Test measurement workflow prompt."""
        response = await prompt_client.get_prompt(
            "measurement-workflow",
            measurement_type="temperature profile",
            ids_involved="core_profiles, equilibrium",
            analysis_goal="transport analysis"
        )

        assert "temperature profile" in response
        assert "core_profiles" in response
        assert "transport analysis" in response
```

### 5.2 Performance Testing

**Purpose**: Ensure tools meet performance requirements

```python
# File: tests/test_performance.py
import pytest
import time
from imas_mcp.server import Server
from imas_mcp.performance.monitor import PerformanceMonitor
from fastmcp.testing import MockContext

class TestPerformance:
    """Performance testing suite."""

    @pytest.fixture
    def server(self):
        """Create server with performance monitoring."""
        server = Server()
        server.performance_monitor = PerformanceMonitor()
        return server

    @pytest.mark.asyncio
    async def test_search_performance_fast_mode(self, server):
        """Test search performance in fast mode."""
        start_time = time.time()

        result = await server.search_imas(
            query="plasma temperature",
            search_mode="fast",
            max_results=10
        )

        execution_time = time.time() - start_time

        # Fast mode should complete in under 0.5 seconds
        assert execution_time < 0.5
        assert result["performance_mode"] == "fast"

    @pytest.mark.asyncio
    async def test_bulk_export_performance(self, server):
        """Test bulk export performance."""
        start_time = time.time()

        result = await server.export_ids(
            ids_list=["core_profiles", "equilibrium"],
            format="raw",
            include_relationships=False
        )

        execution_time = time.time() - start_time

        # Raw export should complete in under 1 second
        assert execution_time < 1.0
        assert result["export_format"] == "raw"

    @pytest.mark.asyncio
    async def test_cache_effectiveness(self, server):
        """Test caching effectiveness."""
        query = "plasma temperature"

        # First call - should miss cache
        start_time = time.time()
        result1 = await server.search_imas(query=query, search_mode="adaptive")
        first_time = time.time() - start_time

        # Second call - should hit cache
        start_time = time.time()
        result2 = await server.search_imas(query=query, search_mode="adaptive")
        second_time = time.time() - start_time

        # Cached call should be significantly faster
        assert second_time < first_time * 0.5
        assert result2.get("cache_hit", False)

    @pytest.mark.asyncio
    async def test_performance_monitoring(self, server):
        """Test performance monitoring functionality."""
        # Execute several operations
        await server.search_imas("plasma", search_mode="fast")
        await server.search_imas("temperature", search_mode="adaptive")
        await server.explain_concept("plasma temperature")

        # Check performance report
        report = server.performance_monitor.get_performance_report()

        assert report["overall_stats"]["total_requests"] >= 3
        assert "tool_breakdown" in report
        assert "search_imas" in report["tool_breakdown"]
```

## Phase 6: Documentation and Deployment (Weeks 15-16)

### 6.1 Comprehensive Documentation

**Purpose**: Document all tools, resources, and prompts

```python
# File: docs/mcp_tools_guide.md
# IMAS MCP Tools Guide

## Tool Performance Modes

### search_imas
- **fast**: Lexical search, <0.5s response time, no AI enhancement
- **adaptive**: Intelligent search selection, <1s response time, conditional AI
- **comprehensive**: Full semantic search, <3s response time, AI enhancement

### export_ids
- **raw**: Pure data export, fastest, no AI enhancement
- **structured**: Organized with relationships, medium speed
- **enhanced**: AI-enhanced insights, slowest, requires AI context

## Resource Usage

### IDS Catalog Resources
- `ids://catalog`: Complete IDS catalog with metadata
- `ids://structure/{ids_name}`: Detailed structure for specific IDS
- `ids://physics-domains`: Physics domain mappings

### Usage Examples Resources
- `examples://search-patterns`: Common search patterns
- `examples://workflows/{type}`: Analysis workflow examples

## Prompt Templates

### Physics Analysis
- `physics-explain`: Generate physics explanations
- `measurement-workflow`: Create measurement workflows
- `cross-ids-analysis`: Analyze cross-IDS relationships

### Code Generation
- `imas-python-code`: Generate Python analysis code
- `data-validation`: Create validation approaches
```

### 6.2 Performance Benchmarks

**Purpose**: Establish performance baselines

```python
# File: benchmarks/performance_benchmarks.py
import asyncio
import time
from typing import Dict, List
from imas_mcp.server import Server

class PerformanceBenchmark:
    """Performance benchmarking suite."""

    def __init__(self):
        self.server = Server()
        self.results: Dict[str, List[float]] = {}

    async def benchmark_search_modes(self, queries: List[str], iterations: int = 10):
        """Benchmark different search modes."""
        modes = ["fast", "adaptive", "comprehensive"]

        for mode in modes:
            times = []

            for _ in range(iterations):
                for query in queries:
                    start_time = time.time()
                    await self.server.search_imas(query=query, search_mode=mode)
                    times.append(time.time() - start_time)

            self.results[f"search_{mode}"] = times

    async def benchmark_export_formats(self, ids_lists: List[List[str]], iterations: int = 5):
        """Benchmark different export formats."""
        formats = ["raw", "structured", "enhanced"]

        for format_type in formats:
            times = []

            for _ in range(iterations):
                for ids_list in ids_lists:
                    start_time = time.time()
                    await self.server.export_ids(
                        ids_list=ids_list,
                        format=format_type
                    )
                    times.append(time.time() - start_time)

            self.results[f"export_{format_type}"] = times

    def generate_report(self) -> Dict[str, Dict[str, float]]:
        """Generate performance benchmark report."""
        report = {}

        for test_name, times in self.results.items():
            report[test_name] = {
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
                "p95_time": sorted(times)[int(len(times) * 0.95)],
                "total_calls": len(times)
            }

        return report

# Performance targets
PERFORMANCE_TARGETS = {
    "search_fast": 0.5,      # seconds
    "search_adaptive": 1.0,   # seconds
    "search_comprehensive": 3.0,  # seconds
    "export_raw": 1.0,       # seconds
    "export_structured": 2.0,  # seconds
    "export_enhanced": 5.0    # seconds
}
```

## Implementation Timeline

| Phase       | Duration    | Key Deliverables                                                                    | Dependencies      |
| ----------- | ----------- | ----------------------------------------------------------------------------------- | ----------------- |
| **Phase 0** | Weeks 1-2   | Comprehensive pytest unit tests, ASV performance monitoring, baseline establishment | Existing codebase |
| **Phase 1** | Weeks 3-5   | Enhanced search_imas, selective AI enhancement, multi-format exports                | Phase 0 complete  |
| **Phase 2** | Weeks 6-8   | MCP resources for IDS data, examples, measurements                                  | Phase 1 complete  |
| **Phase 3** | Weeks 9-10  | MCP prompts for physics analysis, code generation, workflows                        | Phase 2 complete  |
| **Phase 4** | Weeks 11-12 | Caching system, performance monitoring, optimization                                | Phase 3 complete  |
| **Phase 5** | Weeks 13-14 | Comprehensive testing suite, performance validation                                 | Phase 4 complete  |
| **Phase 6** | Weeks 15-16 | Documentation, benchmarking, deployment preparation                                 | Phase 5 complete  |

## Success Metrics

### Performance Metrics

- **Search Fast Mode**: <0.5s response time, >90% cache hit rate
- **Search Adaptive Mode**: <1s response time, intelligent mode selection
- **Export Raw**: <1s for 2 IDS, <2s for 5 IDS
- **Overall Error Rate**: <1% across all tools

### Quality Metrics

- **Test Coverage**: >95% code coverage
- **Tool Suggestion Accuracy**: >80% relevant suggestions
- **Cache Effectiveness**: >70% hit rate for common queries
- **AI Enhancement Value**: Measurable improvement in result quality

### Usage Metrics

- **Tool Adoption**: Usage distribution across all 8 tools
- **Resource Utilization**: Regular access to MCP resources
- **Prompt Usage**: Active use of MCP prompts for workflows

## Risk Mitigation

### Technical Risks

- **Performance Degradation**: Comprehensive monitoring and benchmarking
- **Cache Invalidation**: Intelligent cache management with TTL
- **AI Service Availability**: Graceful degradation when AI unavailable

### Operational Risks

- **Backward Compatibility**: Maintain existing API contracts
- **Resource Consumption**: Monitor memory and CPU usage
- **Scalability**: Design for concurrent user access

This development plan provides a comprehensive roadmap for implementing the critical improvements identified in the tool analysis while expanding capabilities with
