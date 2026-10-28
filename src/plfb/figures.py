"""Summary figure for the README: MAE skill of the best ranked models against the reference, per task.

Usage:
    python -m plfb figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TASK_LABELS = {
    "day_ahead/wind": "Day-ahead wind", "day_ahead/pv": "Day-ahead solar", "day_ahead/demand": "Day-ahead demand",
    "day_ahead/price": "Day-ahead price", "intraday/wind": "Intraday wind", "intraday/pv": "Intraday solar",
    "intraday/demand": "Intraday demand",
}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
GOOD, BAD = "#2a78d6", "#eb6834"
FLOOR = -60.0          # skills below this (naive models) are clipped and labelled


def skill_figure(leaderboard: Path, out: Path, top: int = 4) -> None:
    lb = json.loads(leaderboard.read_text(encoding="utf-8"))
    tasks = [t for t in TASK_LABELS if lb["tasks"].get(t)]
    picked = {}
    for t in tasks:
        ref = "naive_week" if t.endswith("price") else "pse_plan"
        rows = [e for e in lb["tasks"][t] if e.get("ranked") and e["id"] != ref and e.get("skill") is not None]
        picked[t] = rows[:top]
    hi = max(max(e["skill_ci"][1] * 100 for e in rows) for rows in picked.values())
    heights = [len(rows) + 1.2 for rows in picked.values()]
    fig, axes = plt.subplots(len(tasks), 1, figsize=(8.0, 0.28 * sum(heights) + 0.8), facecolor=SURFACE, sharex=True,
                             gridspec_kw={"height_ratios": heights})
    for ax, task in zip(axes, tasks, strict=True):
        rows = picked[task][::-1]
        vals = [e["skill"] * 100 for e in rows]
        shown = [max(v, FLOOR) for v in vals]
        ax.barh(range(len(rows)), shown, color=[GOOD if v >= 0 else BAD for v in vals], height=0.64,
                edgecolor=SURFACE, linewidth=2)
        for i, e in enumerate(rows):
            lo_ci, hi_ci = (max(c * 100, FLOOR) for c in e["skill_ci"])
            ax.plot([lo_ci, hi_ci], [i, i], color=INK2, lw=1)
            ax.text(max(hi_ci, shown[i], 0) + 1.5, i, f"{vals[i]:+.1f}%", va="center", fontsize=8, color=INK2)
        ax.set_yticks(range(len(rows)), [e["name"] for e in rows], fontsize=8.5, color=INK)
        ref = "weekly naive" if task.endswith("price") else "PSE plan"
        ax.set_title(f"{TASK_LABELS[task]}, skill vs {ref}", loc="left", fontsize=9.5, color=INK, pad=3)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_facecolor(SURFACE)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.set_ylim(-0.6, len(rows) - 0.4)
    axes[-1].set_xlim(FLOOR - 2, hi + 12)
    axes[-1].set_xlabel(f"MAE skill against the reference, test year, % (95% interval; bars clipped at {FLOOR:.0f}%)",
                        fontsize=8.5, color=INK2)
    fig.tight_layout(h_pad=0.8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
