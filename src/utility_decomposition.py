# SPDX-License-Identifier: Apache-2.0
"""Split clean utility into its two components, which move in opposite directions.

Clean utility in AgentDojo is not a capability measure, and reading it as one
inverts the conclusion. The utility pass runs over every user task in the
selected suites PLUS every injection goal executed as a standalone user task, and
that second component IS the capability gate. So a model that refuses more
attacker goals scores LOWER clean utility while being better aligned.

  U   user tasks proper, with no attack present. A capability measure.
  G   injection goals run standalone. The capability gate. An ALIGNMENT measure
      once it starts falling, since a refusal and an incapacity score alike.

When the two components move in opposite directions across a lineage, the
aggregate conceals both, and reading the aggregate alone can report an alignment
improvement as a capability regression, or two opposite movements as no change.

This script computes it for any arm from `per_case.csv.gz`, and prints integer
counts beside every rate, for a percentage over 132 tasks does not identify its
numerator and that ambiguity has already carried one error through a full round
of review.

Utility under attack is reported beside U, since that comparison is the one that
validates a security number: a model whose utility collapses under injection was
derailed, and a model whose utility holds was not. **That comparison must use U
and not clean utility**, for its right-hand side contains no injection-goal
tasks and a mixed left-hand side compares different populations.

Usage:
    python src/utility_decomposition.py results_export/<run_id>
    python src/utility_decomposition.py results_export/<run_id> --json-out u_g.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
from pathlib import Path


def _collapse(votes: dict) -> dict:
    """Majority, the convention every other figure in the study uses."""
    return {k: sum(v) * 2 > len(v) for k, v in votes.items()}


def load(export_dir: Path):
    u: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    g: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    ua: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    reps: dict = collections.defaultdict(set)

    with gzip.open(export_dir / "per_case.csv.gz", "rt") as fh:
        for row in csv.DictReader(fh):
            model = row["model"]
            reps[model].add(row["repetition"])
            key = (row["suite"], row["user_task"])
            if row["injection_task"]:
                ua[model][key + (row["injection_task"],)].append(row["utility"] == "1")
            elif row["user_task"].startswith("injection_task"):
                g[model][key].append(row["utility"] == "1")
            else:
                u[model][key].append(row["utility"] == "1")
    return u, g, ua, reps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if not (args.export_dir / "per_case.csv.gz").exists():
        print(f"No per_case.csv.gz under {args.export_dir}")
        return 1

    u, g, ua, reps = load(args.export_dir)
    models = sorted(u)
    manifest = {}
    if (args.export_dir / "manifest.json").exists():
        manifest = json.loads((args.export_dir / "manifest.json").read_text())

    n_u = max((len(u[m]) for m in models), default=0)
    n_g = max((len(g[m]) for m in models), default=0)
    w = max((len(m) for m in models), default=8) + 2

    print("=" * (w + 74))
    print(f"UTILITY DECOMPOSITION   {args.export_dir.name}   "
          f"config_hash {manifest.get('config_hash', '?')}")
    print("=" * (w + 74))
    print(f"\n{n_u} user tasks and {n_g} injection goals, so clean utility runs over "
          f"{n_u + n_g}.")
    print("Majority collapse across repetitions, integer counts beside every rate.\n")
    print(f"  {'model':<{w}}{'U (' + str(n_u) + ')':>13}{'G (' + str(n_g) + ')':>12}"
          f"{'clean (' + str(n_u + n_g) + ')':>16}{'U|attack':>11}{'U loss':>9}{'reps':>6}")
    print("  " + "-" * (w + 67))

    rows = []
    for m in models:
        cu, cg = _collapse(u[m]), _collapse(g[m])
        su, sg = sum(cu.values()), sum(cg.values())
        n = len(cu) + len(cg)
        clean = (su + sg) / n if n else 0.0
        cua = _collapse(ua[m])
        rua = sum(cua.values()) / len(cua) if cua else 0.0
        ur = su / len(cu) if cu else 0.0
        print(f"  {m:<{w}}{f'{su}/{len(cu)} {ur:.1%}':>13}"
              f"{f'{sg}/{len(cg)} {sg / len(cg) if cg else 0:.0%}':>12}"
              f"{f'{su + sg}/{n} {clean:.1%}':>16}{rua:>10.1%}"
              f"{100 * (ur - rua):>+8.1f}p{len(reps[m]):>6}")
        rows.append({"model": m, "u_solved": su, "u_total": len(cu), "u_rate": ur,
                     "g_solved": sg, "g_total": len(cg),
                     "g_rate": sg / len(cg) if cg else 0.0,
                     "clean_solved": su + sg, "clean_total": n, "clean_rate": clean,
                     "utility_under_attack": rua, "u_minus_attacked": ur - rua,
                     "repetitions": len(reps[m])})

    print("\n  'U loss' is U minus utility-under-attack, the derailment measure. Use U")
    print("  on the left of that comparison, never clean utility: the right-hand side")
    print("  contains no injection-goal tasks, so a mixed left-hand side compares two")
    print("  different populations.")

    # The divergence is the finding, so state it rather than leaving it to be read
    # off the table. Opposite-signed movement across a lineage is what makes the
    # aggregate unreportable.
    if len(models) > 1:
        du = 100 * (rows[-1]["u_rate"] - rows[0]["u_rate"])
        dg = 100 * (rows[-1]["g_rate"] - rows[0]["g_rate"])
        dc = 100 * (rows[-1]["clean_rate"] - rows[0]["clean_rate"])
        print(f"\n  First to last model as listed (alphabetical, NOT lineage order):")
        print(f"    U {du:+.1f}p   G {dg:+.1f}p   clean {dc:+.1f}p")
        if du * dg < 0:
            print("    *** U and G move in OPPOSITE directions. The aggregate conceals both;")
            print("        report them separately wherever utility appears.")

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"export": args.export_dir.name, "config_hash": manifest.get("config_hash"),
             "u_total": n_u, "g_total": n_g, "models": rows}, indent=2) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
