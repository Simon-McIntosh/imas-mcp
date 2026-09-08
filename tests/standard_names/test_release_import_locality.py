"""Verify that circular imports between catalog_release and release_notes are avoided."""

import ast
import sys
from pathlib import Path
from typing import Any


def test_catalog_release_no_module_level_release_notes_imports() -> None:
    """Assert that catalog_release.py has no module-level imports from release_notes.

    release_notes imports _github_slug and _run_git from catalog_release at module
    level. If catalog_release also imported from release_notes at module level, it
    would create a circular dependency. This test ensures that all imports from
    release_notes in catalog_release remain function-local.
    """
    catalog_release_path = Path(__file__).parent.parent.parent / "imas_codex" / "standard_names" / "catalog_release.py"
    tree = ast.parse(catalog_release_path.read_text())

    # Collect all module-level ImportFrom nodes that target release_notes
    module_level_imports_from_release_notes = []
    for node in ast.walk(tree):
        # Only check module-level (body of Module), not nested (e.g., inside functions)
        if isinstance(node, ast.Module):
            for item in node.body:
                if isinstance(item, ast.ImportFrom):
                    if item.module and "release_notes" in item.module:
                        module_level_imports_from_release_notes.append(item)

    assert (
        len(module_level_imports_from_release_notes) == 0
    ), (
        f"Found {len(module_level_imports_from_release_notes)} module-level imports from "
        f"release_notes in catalog_release.py. All imports from release_notes must be "
        f"deferred to function-local scope to avoid circular dependency."
    )


def test_both_import_orders_load() -> None:
    """Verify that both import orders work: catalog_release before release_notes and vice versa.

    This ensures that the function-local imports in catalog_release successfully
    break the circular dependency and both modules can be loaded in either order.
    """
    # Get a fresh import state for this test
    modules_to_remove = [
        key for key in sys.modules.keys()
        if "imas_codex.standard_names" in key
    ]
    for key in modules_to_remove:
        del sys.modules[key]

    # Test 1: Import catalog_release first, then release_notes
    try:
        from imas_codex.standard_names import catalog_release as cr1
        from imas_codex.standard_names import release_notes as rn1
        # If we get here, the import order succeeded
        assert hasattr(cr1, "_run_git"), "catalog_release should have _run_git"
        assert hasattr(cr1, "_github_slug"), "catalog_release should have _github_slug"
        assert hasattr(rn1, "build_pr_notes"), "release_notes should have build_pr_notes"
    except ImportError as exc:
        raise AssertionError(f"Failed to import with catalog_release first: {exc}") from exc

    # Clean up for second test
    modules_to_remove = [
        key for key in sys.modules.keys()
        if "imas_codex.standard_names" in key
    ]
    for key in modules_to_remove:
        del sys.modules[key]

    # Test 2: Import release_notes first, then catalog_release
    try:
        from imas_codex.standard_names import release_notes as rn2
        from imas_codex.standard_names import catalog_release as cr2
        # If we get here, the import order succeeded
        assert hasattr(cr2, "_run_git"), "catalog_release should have _run_git"
        assert hasattr(cr2, "_github_slug"), "catalog_release should have _github_slug"
        assert hasattr(rn2, "build_pr_notes"), "release_notes should have build_pr_notes"
    except ImportError as exc:
        raise AssertionError(f"Failed to import with release_notes first: {exc}") from exc
