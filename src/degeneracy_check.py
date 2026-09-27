# SPDX-License-Identifier: Apache-2.0
"""Test whether degenerate trajectories manufacture the measured regression.

On Gemini an empty assistant turn means an unparseable tool call. AgentDojo
scores such a trajectory as a utility failure and, because a model that emits
nothing also injects nothing, almost always as a security PASS. **A model that
degenerates often therefore looks safer than it is**, and `analyze.py` prints a
`degen` column warning as much.

That warning is aimed at reading a security *improvement* as real. The sharper
problem is the mirror image: when degeneracy is asymmetric and favours the BEFORE
model, part of the before model's secure column is not resistance at all, and a
regression measured against it could be an artifact of the older model being
broken rather than safe.

This script answers the question directly: recompute the transition with every
case dropped where the before model produced a degenerate trajectory, and compare.

If the negative-flip rate barely moves, degeneracy is spread evenly across
flipped and unflipped cases and the regression is not built on it. If the rate
falls sharply, the headline is an artifact and must be restated on the clean
subset.

Usage:
    python src/degeneracy_check.py results_export/<run_id> \\
        --before gemini-2.5-flash-lite --after gemini-3.1-flash-lite
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import analyze  # noqa: E402


def load(export_dir: Path, models: list[str], suites: set[str] | None):
    sec: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    deg: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    goal: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    with gzip.open(export_dir / "per_case.csv.gz", "rt") as fh:
        for row in csv.DictReader(fh):
            if row["model"] not in models:
                continue
            if suites and row["suite"] not in suites:
                continue
            if row["injection_task"]:
                if row["secure"] == "":
                    continue
                key = (row["suite"], row["user_task"], row["injection_task"])
                sec[row["model"]][key].append(row["secure"] == "1")
                deg[row["model"]][key].append(row["degenerate"] == "1")
            elif row["user_task"].startswith("injection_task"):
                goal[row["model"]][(row["suite"], row["user_task"])].append(row["utility"] == "1")
    collapse = lambda d: {k: sum(v) * 2 > len(v) for k, v in d.items()}  # noqa: E731
    return (
        {m: collapse(sec[m]) for m in models},
        {m: collapse(deg[m]) for m in models},
        {m: collapse(goal[m]) for m in models},
    )


def rate(sec, before, after, keys):
    neg = sum(1 for k in keys if sec[before][k] and not sec[after][k])
    pos = sum(1 for k in keys if not sec[before][k] and sec[after][k])
    secure_before = sum(1 for k in keys if sec[before][k])
    p, _ = analyze._mcnemar(neg, pos)
    return {
        "cases": len(keys),
        "secure_before": secure_before,
        "negative_flips": neg,
        "positive_flips": pos,
        "nfr": neg / secure_before if secure_before else 0.0,
        "mcnemar_p": p,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--suites", nargs="*", default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    export_dir = args.export_dir if args.export_dir.is_absolute() else Path.cwd() / args.export_dir
    if not (export_dir / "per_case.csv.gz").exists():
        print(f"No per_case.csv.gz under {export_dir}")
        return 1

    before, after = args.before, args.after
    suites = set(args.suites) if args.suites else None
    sec, deg, goal = load(export_dir, [before, after], suites)

    print("=" * 80)
    print(f"DEGENERACY CHECK: {before} -> {after}")
    print(f"{export_dir.name}")
    print("=" * 80)

    print("\nDegeneracy by model, over security cases")
    print(f"  {'model':<26}{'degenerate':>12}{'of':>8}{'share':>9}")
    for m in (before, after):
        n = len(sec[m])
        d = sum(1 for k in sec[m] if deg[m].get(k))
        print(f"  {m:<26}{d:>12}{n:>8}{d / n if n else 0:>8.1%}")
    print("  An asymmetry here favouring the BEFORE model would inflate the")
    print("  measured regression, since a degenerate trajectory scores secure.")

    execu = {k for k, v in goal[before].items() if v} & {k for k, v in goal[after].items() if v}
    results = {}
    for gate, keys in (
        ("raw", set(sec[before]) & set(sec[after])),
        (
            "attack_capability_matched",
            {k for k in set(sec[before]) & set(sec[after]) if (k[0], k[2]) in execu},
        ),
    ):
        clean = {k for k in keys if not deg[before].get(k)}
        full, sub = rate(sec, before, after, keys), rate(sec, before, after, clean)
        results[gate] = {"all_cases": full, "degenerate_before_excluded": sub}
        dropped = len(keys) - len(clean)
        print(f"\n{gate}")
        print(f"  {'subset':<32}{'cases':>7}{'sec-before':>12}{'neg':>6}{'pos':>6}{'NFR':>8}{'p':>11}")
        print("  " + "-" * 82)
        for name, r in (("all cases", full), ("degenerate-before excluded", sub)):
            p = "n/a" if r["mcnemar_p"] is None else f"{r['mcnemar_p']:.2g}"
            print(
                f"  {name:<32}{r['cases']:>7}{r['secure_before']:>12}"
                f"{r['negative_flips']:>6}{r['positive_flips']:>6}{r['nfr']:>7.1%}{p:>11}"
            )
        shift = (sub["nfr"] - full["nfr"]) * 100
        print(f"  dropped {dropped} cases; NFR moves {shift:+.1f} points")

    quoted = results["attack_capability_matched"]
    shift = abs(quoted["degenerate_before_excluded"]["nfr"] - quoted["all_cases"]["nfr"]) * 100
    print()
    if shift < 2:
        print(f"  VERDICT: the quoted rate moves {shift:.1f} points. Degeneracy is spread")
        print("  evenly across flipped and unflipped cases, so the regression is not")
        print("  built on the before model being broken. Report the headline as is")
        print("  and cite this check.")
    else:
        print(f"  VERDICT: the quoted rate moves {shift:.1f} points. That is large enough")
        print("  to matter. Restate the headline on the degeneracy-excluded subset.")

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2, default=float) + "\n")
        print(f"\n  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
