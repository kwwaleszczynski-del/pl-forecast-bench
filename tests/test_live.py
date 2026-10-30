"""The monthly live track: which month is scored, its windows, and that it never touches the frozen test year."""
from __future__ import annotations

import json
from datetime import date

import pytest

from plfb import live
from plfb.config import spec


def test_default_month_waits_for_republished_actuals():
    assert live.default_month(date(2026, 11, 5)) == "2026-10"
    assert live.default_month(date(2026, 11, 4)) == "2026-09"
    assert live.default_month(date(2027, 1, 20)) == "2026-12"


def test_windows_are_a_rolling_year_before_the_month():
    w = live.windows("2026-10")
    assert w["train_start"] == date(2025, 10, 1)
    assert w["month_start"] == date(2026, 10, 1) and w["month_end"] == date(2026, 10, 31)
    assert (w["train_start"] - w["data_start"]).days == live.WARMUP_DAYS
    assert len(live.month_days("2027-02")) == 28


def test_months_inside_v1_and_unfinished_months_are_refused():
    last_v1 = spec().splits["test"][1].strftime("%Y-%m")
    with pytest.raises(ValueError, match="inside benchmark v1"):
        live.check_month(last_v1, date(2027, 1, 1))
    with pytest.raises(ValueError, match="can be scored from"):
        live.check_month("2026-10", date(2026, 11, 3))
    live.check_month("2026-10", date(2026, 11, 5))


def test_index_collects_months_in_order(tmp_path):
    folder = tmp_path / "live"
    folder.mkdir()
    for m in ("2026-11", "2026-10"):
        (folder / f"{m}.json").write_text(json.dumps({"month": m, "models": {}}), encoding="utf-8")
    index = live.build_index(tmp_path)
    assert [m["month"] for m in index["months"]] == ["2026-10", "2026-11"]
    assert json.loads((folder / "index.json").read_text(encoding="utf-8"))["months"][1]["month"] == "2026-11"
