"""Every entry under submissions/ must be complete and consistent, so that pull requests can be checked by CI."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from plfb.config import REPO, spec

SUBS = sorted(p for p in (REPO / "submissions").iterdir() if p.is_dir())
REQUIRED = ("name", "display_name", "kind", "description", "as_of", "code", "authors", "license", "inputs")


@pytest.mark.parametrize("sub", SUBS, ids=[p.name for p in SUBS])
def test_meta(sub: Path):
    meta = json.loads((sub / "meta.json").read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if k not in meta]
    assert not missing, f"{sub.name}: meta.json lacks {missing}"
    assert meta["name"] == sub.name
    tasks = {f"{t}/{v}" for t, v in spec().tasks()}
    assert set(meta.get("not_as_of_tasks", [])) <= tasks


@pytest.mark.parametrize("sub", SUBS, ids=[p.name for p in SUBS])
def test_scores(sub: Path):
    path = sub / "scores_test.json"
    if not path.exists():
        pytest.skip("no test scores yet")
    meta = json.loads((sub / "meta.json").read_text(encoding="utf-8"))
    scores = json.loads(path.read_text(encoding="utf-8"))
    assert scores["split"] == "test" and scores["name"] == sub.name
    assert scores["benchmark_version"] == spec().raw["benchmark"]["version"]
    valid = {f"{t}/{v}" for t, v in spec().tasks()}
    for task, s in scores["tasks"].items():
        assert task in valid
        if not meta.get("reference", False):
            assert s["complete"], f"{sub.name} {task}: incomplete submission"
        for k in ("mae", "rmse", "nmae"):
            assert s[k] >= 0
