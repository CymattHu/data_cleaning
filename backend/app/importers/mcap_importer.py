"""Stub for future ROS 2 Bag / MCAP ingestion.

Real parsing is intentionally not implemented in this interview demo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class McapImporter:
    """Reserved extension point for raw robot bags."""

    supported_suffixes = (".mcap", ".bag", ".db3")

    def can_import(self, path: Path) -> bool:
        return path.suffix.lower() in self.supported_suffixes

    def import_episode(self, path: Path) -> dict[str, Any]:
        # TODO: parse MCAP topics into Episode schema (camera, joint, ft, events)
        raise NotImplementedError(
            "MCAP/ROS Bag import is reserved for a future release. "
            f"Received path: {path}"
        )
