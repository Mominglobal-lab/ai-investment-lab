from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def tmp_path():
    base_dir = ROOT / "scratch_pytest"
    base_dir.mkdir(exist_ok=True)
    temp_dir = base_dir / f"case-{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
