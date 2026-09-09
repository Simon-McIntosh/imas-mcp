"""Closed signed reconstruction for archived standard-name identities."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from imas_codex.standard_names.signed_manifest import (
    SignedManifestAuthorityError,
    SignedManifestConflict,
    apply_signed_manifest,
    signed_payload_sha256,
)

_ADAPTER = "archive-reconstruction"
_MUTATION = "reconstruct-archived-standard-names"
_GUARDS = (
    "signed-exact-identity-manifest",
    "allowlisted-archive-relationships",
    "archive-live-edge-count-parity",
)


class _Transaction:
    def __init__(self, graph: _ArchiveGraph) -> None:
        self.graph = graph
        self.closed = False

    def run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        if "archive-reconstruction-node-state" in cypher:
            node = self.graph.nodes.get(params["id"])
            return (
                [] if node is None else [{"properties": deepcopy(node["properties"])}]
            )
        if "archive-reconstruction-counterpart-state" in cypher:
            return [{"count": int(params["id"] in self.graph.counterparts)}]
        if "archive-reconstruction-create-node" in cypher:
            node = self.graph.nodes.setdefault(
                params["id"], {"properties": deepcopy(params["properties"])}
            )
            return [{"count": int(node["properties"] == params["properties"])}]
        if "archive-reconstruction-create-edge" in cypher:
            relationship_type = re.search(r"\[relationship:([A-Z_]+)\]", cypher).group(
                1
            )
            edge = (
                params["start_id"],
                relationship_type,
                params["end_id"],
                params["properties"],
            )
            if edge not in self.graph.edges:
                self.graph.edges.append(edge)
            return [{"count": 1}]
        if "archive-reconstruction-edge-counts" in cypher:
            return [
                {
                    "relationship_type": relationship_type,
                    "count": sum(
                        1
                        for start, edge_type, end, _ in self.graph.edges
                        if edge_type == relationship_type
                        and params["id"] in {start, end}
                    )
                    + self.graph.extra_counts.get((params["id"], relationship_type), 0),
                }
                for relationship_type in params["types"]
            ]
        raise AssertionError(f"unexpected query: {cypher}")

    def rollback(self) -> None:
        self.closed = True

    def commit(self) -> None:
        self.closed = True


class _Session:
    def __init__(self, graph: _ArchiveGraph) -> None:
        self.graph = graph

    def __enter__(self) -> _Session:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def begin_transaction(self) -> _Transaction:
        return _Transaction(self.graph)


class _ArchiveGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.counterparts = {"unit:eV"}
        self.edges: list[tuple[str, str, str, dict[str, Any]]] = []
        self.extra_counts: dict[tuple[str, str], int] = {}

    def session(self) -> _Session:
        return _Session(self)


def _authority(edges: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    authority = {
        "schema": "imas-codex.archive-reconstruction.v1",
        "operation_id": "reconstruct_archived_standard_name",
        "identities": ["archived_temperature"],
        "nodes": [
            {
                "id": "archived_temperature",
                "properties": {
                    "id": "archived_temperature",
                    "origin": "catalog_edit",
                    "status": "draft",
                    "name_stage": "accepted",
                },
            }
        ],
        "edges": edges
        if edges is not None
        else [
            {
                "owner_id": "archived_temperature",
                "relationship_type": "HAS_UNIT",
                "direction": "outgoing",
                "counterpart_id": "unit:eV",
                "properties": {"source": "archive"},
            }
        ],
    }
    authority["signature"] = {
        "canonicalization": "json-sort-keys-v1",
        "sha256": signed_payload_sha256(authority),
    }
    return authority


def _write_authority(path: Path, authority: dict[str, Any]) -> tuple[str, str]:
    raw = json.dumps(authority, sort_keys=True).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest(), signed_payload_sha256(authority)


def _preview(
    graph: _ArchiveGraph, path: Path, file_hash: str, payload_hash: str
) -> dict[str, Any]:
    return apply_signed_manifest(
        path,
        authority_adapter=_ADAPTER,
        authority_file_sha256=file_hash,
        authority_payload_sha256=payload_hash,
        mutation_kind=_MUTATION,
        guard_set=_GUARDS,
        reason="recover the signed archived identity",
        gc=graph,
    )


def _apply(
    graph: _ArchiveGraph,
    path: Path,
    file_hash: str,
    payload_hash: str,
    manifest_hash: str,
) -> dict[str, Any]:
    return apply_signed_manifest(
        path,
        authority_adapter=_ADAPTER,
        authority_file_sha256=file_hash,
        authority_payload_sha256=payload_hash,
        mutation_kind=_MUTATION,
        guard_set=_GUARDS,
        reason="recover the signed archived identity",
        apply=True,
        manifest_sha256=manifest_hash,
        gc=graph,
    )


def test_signed_archive_reconstruction_creates_identity_and_allowlisted_edges(
    tmp_path: Path,
) -> None:
    graph = _ArchiveGraph()
    path = tmp_path / "authority.json"
    file_hash, payload_hash = _write_authority(path, _authority())

    preview = _preview(graph, path, file_hash, payload_hash)
    applied = _apply(graph, path, file_hash, payload_hash, preview["manifest_sha256"])

    assert preview["outcome"] == "would_apply"
    assert applied["outcome"] == "applied"
    assert graph.nodes["archived_temperature"]["properties"]["origin"] == "pipeline"
    assert graph.edges == [
        ("archived_temperature", "HAS_UNIT", "unit:eV", {"source": "archive"})
    ]


def test_archive_reconstruction_refuses_relationship_outside_registry(
    tmp_path: Path,
) -> None:
    path = tmp_path / "authority.json"
    file_hash, payload_hash = _write_authority(
        path,
        _authority(
            [
                {
                    "owner_id": "archived_temperature",
                    "relationship_type": "ARBITRARY_EDGE",
                    "direction": "outgoing",
                    "counterpart_id": "unit:eV",
                    "properties": {},
                }
            ]
        ),
    )

    with pytest.raises(
        SignedManifestAuthorityError, match="outside the reconstruction registry"
    ):
        _preview(_ArchiveGraph(), path, file_hash, payload_hash)


def test_archive_reconstruction_refuses_missing_counterpart(tmp_path: Path) -> None:
    graph = _ArchiveGraph()
    graph.counterparts.clear()
    path = tmp_path / "authority.json"
    file_hash, payload_hash = _write_authority(path, _authority())

    preview = _preview(graph, path, file_hash, payload_hash)

    assert preview["outcome"] == "refused"
    assert "neither live nor included" in preview["refusals"][0]["reason"]


def test_archive_reconstruction_refuses_nonidentical_existing_node(
    tmp_path: Path,
) -> None:
    graph = _ArchiveGraph()
    graph.nodes["archived_temperature"] = {
        "properties": {
            "id": "archived_temperature",
            "origin": "pipeline",
            "status": "active",
        }
    }
    path = tmp_path / "authority.json"
    file_hash, payload_hash = _write_authority(path, _authority())

    preview = _preview(graph, path, file_hash, payload_hash)

    assert preview["outcome"] == "refused"
    assert (
        "differs from signed reconstruction state" in preview["refusals"][0]["reason"]
    )


def test_archive_reconstruction_refuses_archive_live_edge_count_mismatch(
    tmp_path: Path,
) -> None:
    graph = _ArchiveGraph()
    graph.extra_counts[("archived_temperature", "HAS_UNIT")] = 1
    path = tmp_path / "authority.json"
    file_hash, payload_hash = _write_authority(path, _authority())
    preview = _preview(graph, path, file_hash, payload_hash)

    with pytest.raises(SignedManifestConflict, match="relationship counts differ"):
        _apply(graph, path, file_hash, payload_hash, preview["manifest_sha256"])
