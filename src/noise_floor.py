# SPDX-License-Identifier: Apache-2.0
"""Measure each model's noise floor by pairing it against itself.

A negative-flip rate is only interpretable against the rate at which the *same
model* flips cases when simply rerun. That rate is the noise floor, and it is
reported per model because **it cannot be inferred from a model's attack success
rate**: two models with nearly the same attack success rate can carry very
different floors.

A floor that is reasoned about rather than computed, or borrowed from a pooled
figure, can misstate how far a transition clears it even when the conclusion
survives. Hence this script.

Two statistics per model, because two conventions are in use and conflating them
is how the above happened:

  range        min and max over all ordered repetition pairs. The honest summary,
               since it shows how much the estimate itself moves.
  rep1 -> rep2 the single ordered pair that `compare_arms.noise_floor` reports.
               Quoted here so the two agree by construction rather than by luck.

A flip counts when a case is secure in the first repetition and insecure in the
second. AgentDojo's `security` field is True when the ATTACK SUCCEEDED, and the
exporter has already inverted it, so `secure == 1` here means the attack failed.

Usage:
    python src/noise_floor.py results_export/<run_id>
    python src/noise_floor.py results_export/<run_id> --suites slack travel banking
    python src/noise_floor.py results_export/<run_id> --json-out floors.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import itertools
import json
from pathlib import Path

SRC = Path(__file__).resolve().parent
EXPORT_ROOT = SRC.parent / "results_export"


def load(export_dir: Path, suites: set[str] | None) -> dict[str, dict[str, dict[tuple, bool]]]:
    """model -> repetition -> case -> secure.

    Security cases only. A row with an empty `secure` field is a crashed or
    excluded trace and is dropped rather than scored.
    """
    per: dict[str, dict[str, dict[tuple, bool]]] = collections.defaultdict(
        lambda: collections.defaultdict(dict)
    )
    with gzip.open(export_dir / "per_case.csv.gz", "rt") as fh:
        for row in csv.DictReader(fh):
            if not row["injection_task"] or row["secure"] == "":
                continue
            if suites and row["suite"] not in suites:
                continue
            key = (row["suite"], row["user_task"], row["injection_task"])
            per[row["model"]][row["repetition"]][key] = row["secure"] == "1"
    return per


def floors(reps: dict[str, dict[tuple, bool]]) -> dict:
    """Spurious negative flips for every ordered pair of repetitions."""
    names = sorted(reps)
    if len(names) < 2:
        return {}
    shared = set.intersection(*(set(v) for v in reps.values()))
    pairs = {
        f"{a}->{b}": sum(1 for k in shared if reps[a][k] and not reps[b][k])
        for a, b in itertools.permutations(names, 2)
    }
    counts = list(pairs.values())
    a, b = names[0], names[1]
    asr_a = 1 - sum(1 for k in shared if reps[a][k]) / len(shared) if shared else 0.0
    asr_b = 1 - sum(1 for k in shared if reps[b][k]) / len(shared) if shared else 0.0
    neg = pairs[f"{a}->{b}"]
    pos = pairs[f"{b}->{a}"]
    return {
        "cases": len(shared),
        "repetitions": len(names),
        "asr_rep1": round(asr_a, 4),
        "asr_rep2": round(asr_b, 4),
        "delta_asr_pt": round((asr_b - asr_a) * 100, 2),
        # Cases that changed outcome in EITHER direction between two runs of the
        # same model. This is the quantity aggregate ASR cannot see: it nets to
        # nearly nothing while a large share of cases move.
        "churn": neg + pos,
        "churn_share": round((neg + pos) / len(shared), 4) if shared else 0.0,
        "pairs": pairs,
        "min": min(counts),
        "max": max(counts),
        "rep1_to_rep2": neg,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("export_dir", type=Path, help="A results_export/<run_id> directory.")
    parser.add_argument(
        "--suites",
        nargs="*",
        default=None,
        help="Restrict to these suites. Default: all suites present in the export.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    export_dir = args.export_dir
    if not export_dir.is_absolute():
        export_dir = Path.cwd() / export_dir
    if not (export_dir / "per_case.csv.gz").exists():
        # Tolerate being handed the run directory instead of the export.
        alt = EXPORT_ROOT / export_dir.name.replace("google_", "").replace("baseten_", "")
        if (alt / "per_case.csv.gz").exists():
            export_dir = alt
        else:
            print(f"No per_case.csv.gz under {export_dir}")
            return 1

    per = load(export_dir, set(args.suites) if args.suites else None)
    if not per:
        print("No security cases found.")
        return 1

    scope = " ".join(sorted(args.suites)) if args.suites else "all suites"
    print("=" * 78)
    print(f"NOISE FLOOR: {export_dir.name}   ({scope})")
    print("=" * 78)
    print("Each model paired against ITSELF across repetitions. A transition whose")
    print("negative flips do not exceed its BEFORE-model's floor is not")
    print("distinguishable from rerunning that model.")
    print()

    results = {}
    rows = []
    for model, reps in per.items():
        f = floors(reps)
        if not f:
            print(f"  {model}: only {len(reps)} repetition, no floor measurable")
            continue
        results[model] = f
        rows.append((f["asr_rep1"], model, f))

    width = max(len(m) for m in results) if results else 30
    print(
        f"  {'model':<{width}}{'ASR':>8}{'cases':>7}{'floor range':>13}"
        f"{'r1->r2':>8}{'dASR pt':>9}{'churn':>7}{'churn %':>9}"
    )
    print("  " + "-" * (width + 61))
    for asr, model, f in sorted(rows, reverse=True):
        span = f"{f['min']}" if f["min"] == f["max"] else f"{f['min']}-{f['max']}"
        print(
            f"  {model:<{width}}{asr:>7.1%}{f['cases']:>7}{span:>13}"
            f"{f['rep1_to_rep2']:>8}{f['delta_asr_pt']:>+9.1f}{f['churn']:>7}{f['churn_share']:>8.1%}"
        )

    spread = [f for _, _, f in rows]
    if spread:
        lo = min(f["min"] for f in spread)
        hi = max(f["max"] for f in spread)
        print()
        print(f"  Floors span {lo} to {hi} flips across models in this export.")
        print("  Judge every transition against its own before-model, never against")
        print("  a pooled figure and never against a guess from the model's rate.")
        worst = max(spread, key=lambda f: f["churn_share"])
        if worst["churn"]:
            print()
            print("  The last three columns are the case for the whole method. Rerunning")
            print("  ONE model changes the outcome on `churn` cases while `dASR` barely")
            print(f"  moves: here up to {worst['churn']} cases, {worst['churn_share']:.1%} of the set,")
            print("  against an aggregate shift of well under a point. Aggregate attack")
            print("  success rate cannot see that movement, and a single-repetition")
            print("  leaderboard reports whichever draw it happened to take.")

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\n  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
