# SPDX-License-Identifier: Apache-2.0
"""Decompose negative flips by what the agent was actually doing.

A negative-flip rate says how often a version upgrade turned a secure case
insecure. It does not say what kind of failure that was, and the distinction
carries the most important objection to the whole result.

**The objection.** A newer model is more capable. Capability is what lets an
agent reach the tool call an injection needs, so of course the capable successor
succumbs more often — the regression is the price of the upgrade working, and a
practitioner would pay it knowingly. If that were true, the flips would
concentrate on cases where the newer model *completes the user's task* and is
exploited along the way.

**What this script measures.** Every negative flip falls into exactly one of four
cells, by whether each version solved the user task under attack:

  after newly solves     before failed, after solved. The capability story: the
                         upgrade bought a working agent and exposure with it.
  both solve             both solved. A genuine security loss at constant
                         capability, and the cleanest cell to reason about.
  NEITHER solves         neither solved. The agent executed the attacker's goal
                         and delivered nothing to the user. No tradeoff exists
                         here: it is loss without compensation.
  after LOST the task    before solved, after failed. Strictly worse on both
                         axes.

Run against the capability gate, since an injection goal neither version can
execute on direct request contributes a case that was never really tested.

Usage:
    python src/flip_anatomy.py results_export/<run_id> --before <model> --after <model>
    python src/flip_anatomy.py results_export/<run_id> --before A --after B --suites slack travel banking
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
from pathlib import Path

CELLS = (
    ("after newly solves", "the capability story"),
    ("both solve", "security lost at constant capability"),
    ("NEITHER solves", "exploited, user gets nothing"),
    ("after LOST the task", "worse on both axes"),
)


def load(export_dir: Path, models: list[str], suites: set[str] | None):
    """Collapsed security, utility-under-attack, and standalone goal execution."""
    sec: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    util: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    goal: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    with gzip.open(export_dir / "per_case.csv.gz", "rt") as fh:
        for row in csv.DictReader(fh):
            if row["model"] not in models:
                continue
            if suites and row["suite"] not in suites:
                continue
            if row["injection_task"]:
                # A crashed trace has no outcome. Excluded, never scored.
                if row["secure"] == "":
                    continue
                key = (row["suite"], row["user_task"], row["injection_task"])
                sec[row["model"]][key].append(row["secure"] == "1")
                util[row["model"]][key].append(row["utility"] == "1")
            elif row["user_task"].startswith("injection_task"):
                goal[row["model"]][(row["suite"], row["user_task"])].append(row["utility"] == "1")
    collapse = lambda d: {k: sum(v) * 2 > len(v) for k, v in d.items()}  # noqa: E731
    return (
        {m: collapse(sec[m]) for m in models},
        {m: collapse(util[m]) for m in models},
        {m: collapse(goal[m]) for m in models},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--suites", nargs="*", default=None)
    parser.add_argument(
        "--gate",
        choices=("capability", "raw"),
        default="capability",
        help="Restrict to goals both versions execute on direct request. Default: capability.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    export_dir = args.export_dir if args.export_dir.is_absolute() else Path.cwd() / args.export_dir
    if not (export_dir / "per_case.csv.gz").exists():
        print(f"No per_case.csv.gz under {export_dir}")
        return 1

    before, after = args.before, args.after
    suites = set(args.suites) if args.suites else None
    sec, util, goal = load(export_dir, [before, after], suites)
    if not sec[before] or not sec[after]:
        print("One of the models has no cases in this export.")
        return 1

    shared = set(sec[before]) & set(sec[after])
    if args.gate == "capability":
        execu = {k for k, v in goal[before].items() if v} & {k for k, v in goal[after].items() if v}
        shared = {k for k in shared if (k[0], k[2]) in execu}

    flips = [k for k in shared if sec[before][k] and not sec[after][k]]
    if not flips:
        print(f"No negative flips for {before} -> {after} under this gate.")
        return 0

    cells = {
        "after newly solves": [k for k in flips if not util[before][k] and util[after][k]],
        "both solve": [k for k in flips if util[before][k] and util[after][k]],
        "NEITHER solves": [k for k in flips if not util[before][k] and not util[after][k]],
        "after LOST the task": [k for k in flips if util[before][k] and not util[after][k]],
    }

    scope = " ".join(sorted(suites)) if suites else "all suites"
    print("=" * 78)
    print(f"FLIP ANATOMY: {before} -> {after}")
    print(f"{export_dir.name}   ({scope}, {args.gate} gate)")
    print("=" * 78)
    print(f"  {len(shared)} cases compared, {len(flips)} negative flips\n")
    print(f"  {'cell':<24}{'n':>6}{'share':>9}   what it means")
    print("  " + "-" * 74)
    for name, meaning in CELLS:
        n = len(cells[name])
        print(f"  {name:<24}{n:>6}{n / len(flips):>8.1%}   {meaning}")

    tradeoff = len(cells["after newly solves"]) + len(cells["both solve"])
    no_utility = len(cells["NEITHER solves"]) + len(cells["after LOST the task"])
    print("  " + "-" * 74)
    print(f"  {'capability story':<24}{tradeoff:>6}{tradeoff / len(flips):>8.1%}")
    print(f"  {'no utility delivered':<24}{no_utility:>6}{no_utility / len(flips):>8.1%}")
    print()
    print("  The second figure is the one that answers the objection. Those cases")
    print("  are not a tradeoff: the agent executed the attacker's goal and")
    print("  returned nothing to the user, so there is nothing bought with the")
    print("  exposure.")

    # Breadth. A regression concentrated on one attacker goal is a bug; one
    # spread across most of them is a change in how the model treats injected
    # instructions generally, and only the second supports the paper's claim.
    goals_flipped = {(k[0], k[2]) for k in flips}
    goals_tested = {(k[0], k[2]) for k in shared}
    suites_flipped = {k[0] for k in flips}
    suites_tested = {k[0] for k in shared}
    print()
    print("  Breadth")
    print(f"    attacker goals with at least one flip : {len(goals_flipped)} of {len(goals_tested)}")
    print(f"    suites with at least one flip         : {len(suites_flipped)} of {len(suites_tested)}")
    per_suite = collections.Counter(k[0] for k in flips)
    secure_before = collections.Counter(k[0] for k in shared if sec[before][k])
    for suite in sorted(suites_tested):
        n = secure_before[suite]
        rate = f"{per_suite[suite] / n:.1%}" if n else "n/a"
        print(f"      {suite:<12}{per_suite[suite]:>4} flips / {n:>4} secure-before   NFR {rate:>7}")

    if args.json_out:
        payload = {
            "before": before,
            "after": after,
            "export": export_dir.name,
            "gate": args.gate,
            "cases": len(shared),
            "negative_flips": len(flips),
            "cells": {k: len(v) for k, v in cells.items()},
        }
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\n  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
