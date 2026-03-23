from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_project_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    root_candidate = PROJECT_ROOT / candidate
    if root_candidate.exists():
        return root_candidate.resolve()
    return root_candidate
