# SPDX-License-Identifier: Apache-2.0
"""Put the noise floor and the transition on one estimand, which they are not.

**This is a defect fix, not a strengthening.** The regression decision rule requires
a transition's negative flips to exceed the predecessor's noise floor. As the study
computes them, those two quantities live on different scales:

  the transition   collapsed. Each model's per-case outcome is a majority vote over
                   its repetitions, and flips are counted between the two collapsed
                   maps. One number per transition.
  the floor        uncollapsed. Flips are counted between two INDIVIDUAL repetitions
                   of one model, over every ordered pair. A distribution.

A collapsed count is not commensurable with an uncollapsed one, and the direction of
the incommensurability is not even constant: at two repetitions the collapsed insecure
set is the UNION of the two repetitions' failures, which inflates the transition count
relative to any single-repetition comparison, while at three or more the majority rule
shrinks it. So the rule compares a union against a sample on some arms and a
majority against a sample on others.

The fix implemented here is to put both on the **per-repetition** scale, which is
available on every arm. For a transition from A to B with R repetitions each, the
flip count is computed for every ordered pair of repetitions, giving R^2 values; the
floor for A is computed over its R(R-1) ordered self-pairs. Both are then counts of
flips between two single samples, and the comparison is like with like.

The collapsed figure is still printed, for it is what the papers quote and the point
of this script is to show how far it sits from the like-for-like distribution.

The alternative fix, collapsing the floor as well by splitting repetitions into two
halves and collapsing each, needs at least four repetitions and only one model in the
study has them. It is noted here so the choice is visible rather than implicit.

Usage:
    python src/estimand_check.py results_export/<run_id> --lineage A B [C ...]
    python src/estimand_check.py results_export/<run_id> --all-consecutive
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import itertools
import json
import statistics
from pathlib import Path


def _rep_index(label: str) -> int:
    digits = "".join(c for c in label if c.isdigit())
    return int(digits) if digits else 0


def load(export_dir: Path, max_reps: int | None = None):
    """Per-repetition security maps. Nothing is collapsed on the way in."""
    per: dict = collections.defaultdict(lambda: collections.defaultdict(dict))
    with gzip.open(export_dir / "per_case.csv.gz", "rt") as fh:
        for row in csv.DictReader(fh):
            if not row["injection_task"] or row["secure"] == "":
                continue
            rep = row["repetition"]
            if max_reps is not None and _rep_index(rep) > max_reps:
                continue
            key = (row["suite"], row["user_task"], row["injection_task"])
            per[row["model"]][rep][key] = row["secure"] == "1"
    return per


def collapse(reps: dict) -> dict:
    votes: dict = collections.defaultdict(list)
    for rep in reps.values():
        for k, v in rep.items():
            votes[k].append(v)
    return {k: sum(v) * 2 > len(v) for k, v in votes.items()}


def flips(a: dict, b: dict) -> int:
    """Secure under a, insecure under b, over the cases both hold."""
    shared = set(a) & set(b)
    return sum(1 for k in shared if a[k] and not b[k])


def pairwise(before: dict, after: dict, same_model: bool) -> list[int]:
    """Flip counts over ordered repetition pairs.

    For a transition every (i, j) is admissible including i == j, since the two
    repetitions come from different models. For a floor the diagonal is excluded,
    a repetition compared against itself yielding zero by construction.
    """
    out = []
    for ra, rb in itertools.product(sorted(before, key=_rep_index),
                                    sorted(after, key=_rep_index)):
        if same_model and ra == rb:
            continue
        out.append(flips(before[ra], after[rb]))
    return out


def _fmt(vals: list[int]) -> str:
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]}"
    return f"{min(vals)}-{max(vals)} (med {statistics.median(vals):g}, n={len(vals)})"


def report(export_dir: Path, chains: list[list[str]], max_reps: int | None):
    per = load(export_dir, max_reps)
    manifest = {}
    if (export_dir / "manifest.json").exists():
        manifest = json.loads((export_dir / "manifest.json").read_text())

    print("=" * 100)
    print(f"ESTIMAND CHECK   {export_dir.name}   config_hash {manifest.get('config_hash')}")
    print("=" * 100)
    print("\nCollapsed flips are what the papers quote. The per-repetition columns are")
    print("the like-for-like comparison: both sides count flips between two single")
    print("samples. A verdict that changes between the two columns was resting on the")
    print("estimand rather than on the data.\n")

    rows = []
    w = max((len(f"{a} -> {b}") for c in chains for a, b in zip(c, c[1:])), default=20) + 2
    print(f"  {'transition':<{w}}{'collapsed':>10}{'per-rep flips':>26}{'per-rep floor':>26}{'verdict':>10}")
    print("  " + "-" * (w + 72))
    for chain in chains:
        chain = [m for m in chain if m in per]
        for a, b in zip(chain, chain[1:]):
            coll = flips(collapse(per[a]), collapse(per[b]))
            tr = pairwise(per[a], per[b], same_model=False)
            fl = pairwise(per[a], per[a], same_model=True)
            # A transition clears its floor on the like-for-like scale when its
            # WORST pairing still exceeds the floor's BEST, which is the
            # conservative reading and the one a referee will apply.
            if not tr or not fl:
                verdict = "n/a"
            elif max(tr) == 0:
                # Not a regression candidate at all. Calling this "rejected by the
                # floor" would be a category error: there is nothing to reject.
                verdict = "no cand."
            elif min(tr) > max(fl):
                verdict = "clears"
            elif max(tr) <= min(fl):
                verdict = "REJECTED"
            else:
                verdict = "overlaps"
            print(f"  {a + ' -> ' + b:<{w}}{coll:>10}{_fmt(tr):>26}{_fmt(fl):>26}{verdict:>10}")
            rows.append({"before": a, "after": b, "collapsed_flips": coll,
                         "per_rep_flips": tr, "per_rep_floor": fl, "verdict": verdict})

    print("\n  'clears' means the transition's WORST repetition pairing still exceeds the")
    print("  floor's BEST, which is the conservative reading. 'overlaps' means the two")
    print("  distributions intersect and the collapsed verdict cannot be trusted on its")
    print("  own. Report the interval, not the point.")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--lineage", nargs="+", action="append", default=None,
                        help="An ordered chain of model ids. Repeatable.")
    parser.add_argument("--all-consecutive", action="store_true",
                        help="Every alphabetically adjacent pair, for a quick look.")
    parser.add_argument("--max-reps", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if not (args.export_dir / "per_case.csv.gz").exists():
        print(f"No per_case.csv.gz under {args.export_dir}")
        return 1

    chains = args.lineage or []
    if args.all_consecutive or not chains:
        models = sorted(load(args.export_dir, args.max_reps))
        chains = [models]
    rows = report(args.export_dir, chains, args.max_reps)

    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
