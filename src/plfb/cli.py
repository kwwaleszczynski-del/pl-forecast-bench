"""Command line interface: ``python -m plfb <command>`` (or ``plfb <command>``).

Usage:
    python -m plfb nwp --start 2024-06-13 --end 2026-09-30     # archived ECMWF runs (Open-Meteo)
    python -m plfb pse                                         # PSE reports for the benchmark period
    python -m plfb build                                       # canonical tables + checksums
    python -m plfb verify                                      # compare your download with benchmark/checksums.json
    python -m plfb baselines --jobs 2                          # forecasts of the reference models
    python -m plfb evaluate --forecast my.parquet --name my-model --out submissions/my-model
    python -m plfb leaderboard                                 # docs/leaderboard.json from submissions/*/scores.json
    python -m plfb figures                                     # docs/img/skill.png
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

from plfb import config


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="plfb", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", help="data directory (default: $PLFB_DATA or ./data)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("nwp", help="download archived ECMWF IFS runs from Open-Meteo")
    s.add_argument("--start", type=date.fromisoformat, default=date(2024, 6, 13))
    s.add_argument("--end", type=date.fromisoformat, default=date(2026, 9, 30))
    s.add_argument("--pause", type=float, default=7.0, help="seconds between requests (each costs 8 API units)")

    s = sub.add_parser("pse", help="download the PSE reports (through pse-opendata)")
    s.add_argument("--end", type=date.fromisoformat, default=date(2026, 10, 1))

    sub.add_parser("build", help="build the canonical tables and write their checksums")
    s = sub.add_parser("verify", help="compare the local tables with benchmark/checksums.json")
    s.add_argument("--checksums", type=Path, default=config.BENCH_DIR / "checksums.json")

    s = sub.add_parser("baselines", help="run the reference models and write their forecasts")
    s.add_argument("--models", nargs="*", help="subset of models (default: all non-foundation models)")
    s.add_argument("--jobs", type=int, default=2)
    s.add_argument("--splits", nargs="*", default=["test"], choices=["train", "val", "test"])
    s.add_argument("--out", type=Path, default=None, help="forecast directory (default: <data-dir>/forecasts)")

    s = sub.add_parser("foundation", help="zero-shot foundation models (needs the [foundation] extra)")
    s.add_argument("--models", nargs="*")
    s.add_argument("--device", default="cpu")
    s.add_argument("--splits", nargs="*", default=["test"], choices=["val", "test"])
    s.add_argument("--out", type=Path, default=None)

    s = sub.add_parser("evaluate", help="score a forecast file")
    s.add_argument("--forecast", type=Path, required=True, nargs="+")
    s.add_argument("--name", required=True)
    s.add_argument("--split", default="test", choices=["val", "test"])
    s.add_argument("--out", type=Path, help="directory for scores.json (default: print a summary only)")
    s.add_argument("--meta", type=Path, help="meta.json of the submission to copy next to the scores")

    s = sub.add_parser("leaderboard", help="collect submissions/*/scores.json into the site data")
    s.add_argument("--submissions", type=Path, default=config.REPO / "submissions")
    s.add_argument("--docs", type=Path, default=config.REPO / "docs")

    s = sub.add_parser("figures", help="summary figure of the leaderboard for the README")
    s.add_argument("--docs", type=Path, default=config.REPO / "docs")

    args = p.parse_args(argv)
    _setup_logging(args.verbose)
    root = config.data_dir(args.data_dir)

    if args.cmd == "nwp":
        from plfb import nwp
        try:
            n = nwp.download(root, args.start, args.end, pause=args.pause)
        except nwp.DailyLimitError as e:
            logging.error("Open-Meteo daily limit reached (%s); run again after the reset", e)
            return 2
        logging.info("downloaded %d runs", n)
    elif args.cmd == "pse":
        from plfb import pse_data
        pse_data.download(root, end=args.end)
    elif args.cmd == "build":
        from plfb import dataset
        dataset.build(root)
    elif args.cmd == "verify":
        from plfb import dataset
        return 0 if dataset.verify(root, args.checksums) else 1
    elif args.cmd == "baselines":
        os.environ.setdefault("OMP_NUM_THREADS", str(args.jobs))
        from plfb.baselines import run
        run.run_all(root, args.out or root / "forecasts", models=args.models, splits=args.splits, jobs=args.jobs)
    elif args.cmd == "foundation":
        from plfb.baselines import foundation
        foundation.run(root, args.out or root / "forecasts", models=args.models, device=args.device,
                       splits=args.splits)
    elif args.cmd == "evaluate":
        from plfb import evaluate
        evaluate.cli(root, args.forecast, args.name, args.split, args.out, args.meta)
    elif args.cmd == "leaderboard":
        from plfb import leaderboard
        leaderboard.build(args.submissions, args.docs)
    elif args.cmd == "figures":
        from plfb import figures
        figures.skill_figure(args.docs / "leaderboard.json", args.docs / "img" / "skill.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
