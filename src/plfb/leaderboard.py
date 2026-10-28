"""Collect ``submissions/<name>/{meta.json, scores_test.json}`` into ``docs/leaderboard.json`` for the site.

A submission is ranked in a task if its meta says ``"as_of": true`` and does not list the task in
``not_as_of_tasks``; references that use information published after the origin (the PSE plan day-ahead) are shown
but not ranked.

Usage:
    python -m plfb leaderboard
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from plfb.config import spec

log = logging.getLogger(__name__)

KEEP = ("mae", "rmse", "nmae", "bias", "pinball", "crps", "coverage80", "width80", "imbalance_cost_pln_mwh",
        "rows", "rows_expected", "complete", "by_lead_hour", "by_hour", "by_month")


def _entry(meta: dict, task: str, s: dict) -> dict:
    e = {
        "id": meta["name"], "name": meta.get("display_name", meta["name"]), "kind": meta.get("kind", ""),
        "as_of": bool(meta.get("as_of", True)) and task not in meta.get("not_as_of_tasks", []),
        "reference": bool(meta.get("reference", False)),
        "description": meta.get("description", ""), "inputs": meta.get("inputs", []), "code": meta.get("code", ""),
        "authors": meta.get("authors", []),
    }
    e["ranked"] = e["as_of"]
    e.update({k: s[k] for k in KEEP if k in s})
    if "reference" in s:
        r = s["reference"]
        e["ref_name"] = r["name"]
        e["skill"] = r["skill_mae"]
        e["skill_ci"] = r["skill_ci95"]
        e["ref_rows"] = r["rows"]
    if "battery" in s:
        e["battery_share"] = s["battery"]["share_of_perfect"]
        e["battery_pln_mw_year"] = s["battery"]["pln_per_mw_year"]
    return e


def build(submissions: Path, docs: Path, split: str = "test") -> dict:
    tasks: dict[str, list[dict]] = {f"{t}/{v}": [] for t, v in spec().tasks()}
    for d in sorted(p for p in submissions.iterdir() if p.is_dir()):
        meta_p, sc_p = d / "meta.json", d / f"scores_{split}.json"
        if not (meta_p.exists() and sc_p.exists()):
            log.warning("%s: no meta.json or scores_%s.json, skipped", d.name, split)
            continue
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        scores = json.loads(sc_p.read_text(encoding="utf-8"))
        for task, s in scores["tasks"].items():
            tasks[task].append(_entry(meta, task, s))
    for rows in tasks.values():
        rows.sort(key=lambda e: (not e["ranked"], e["mae"]))
        rank = 0
        for e in rows:
            if e["ranked"]:
                rank += 1
                e["rank"] = rank
    lo, hi = spec().splits[split]
    out = {
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark_version": spec().raw["benchmark"]["version"],
        "split": split, "period": [lo.isoformat(), hi.isoformat()],
        "quantiles": spec().quantiles,
        "tasks": tasks,
    }
    docs.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, indent=1, ensure_ascii=False) + "\n"
    (docs / "leaderboard.json").write_text(text, encoding="utf-8", newline="\n")
    log.info("leaderboard: %s", ", ".join(f"{k} ({len(v)})" for k, v in tasks.items()))
    return out
