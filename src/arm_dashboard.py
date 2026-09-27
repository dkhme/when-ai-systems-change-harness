# SPDX-License-Identifier: Apache-2.0
"""Per-model capability gate and every pairwise comparison, gated and ungated.

Complements `analyze.py` rather than replacing it. That script prints per-model
attack success with confidence intervals and pairs CONSECUTIVE versions within
each registry lineage, which is the right default when the arm is a lineage.

The OpenAI arm is not a lineage, and two things fall through the gap.

**The capability gate is not printed anywhere.** AgentDojo runs each injection
goal as a standalone user task, so the number of goals a model executes on
direct request is measured on every pass and simply never surfaced. On an arm
where attack success may be near zero, that number decides whether a zero means
resistance or incapacity, which is the difference between a finding and an
instrument with no range left.

**A model belongs to one lineage in the registry, so a model that sits in two
structures is only paired in one.** `gpt-5.6-sol` is the successor in the
flagship pair and simultaneously the top of the 5.6 tier ladder; the registry
files it under the former, so `analyze.py` never pairs `terra -> sol`. This
prints all ordered pairs and lets the reader decide which are version
transitions and which are tier steps.

**A pairwise number here is arithmetic, not interpretation.** Negative flips
between two members of the SAME generation measure a capability difference
between siblings, not a regression across a version boundary. The arithmetic is
identical and the meaning is not. Label the ladder rows as tier steps wherever
they are quoted.

    python src/arm_dashboard.py results_export/<run_id>

Reads `per_case.csv.gz`, so it works on any arm's export.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import itertools
import json
import math
from pathlib import Path


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, matching the convention the other scripts use."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar on the discordant pairs.

    Under the null the b + c discordant cases split as a fair coin. With c = 0
    this reduces to the closed form 2 * 0.5**b.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) * 0.5**n
    return min(1.0, 2 * tail)


def _rep_index(label: str) -> int:
    """'rep3' -> 3. Repetition labels are the only ordering available."""
    digits = "".join(c for c in label if c.isdigit())
    return int(digits) if digits else 0


def load(export_dir: Path, max_reps: int | None = None):
    """Collapse repetitions by majority vote, exactly as flip_anatomy does.

    `max_reps` keeps only repetitions 1..max_reps. It exists because a model can
    legitimately carry more repetitions than its neighbours -- gpt-5.4-nano was
    extended to four while the rest of the GPT arm stayed at two -- and the
    majority threshold is NOT the same at both. At two repetitions the test
    `sum(votes) * 2 > len(votes)` demands both; at four it demands three of four.
    Comparing a two-repetition union against a four-repetition majority
    manufactures a difference out of the convention. Level the counts before any
    cross-model table.
    """
    sec: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    util: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    goal: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    clean: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    degen: dict = collections.Counter()
    total: dict = collections.Counter()
    reps: dict = collections.defaultdict(set)

    with gzip.open(export_dir / "per_case.csv.gz", "rt") as fh:
        for row in csv.DictReader(fh):
            model = row["model"]
            if max_reps is not None and _rep_index(row["repetition"]) > max_reps:
                continue
            reps[model].add(row["repetition"])
            total[model] += 1
            degen[model] += int(row["degenerate"] or 0)
            if row["injection_task"]:
                if row["secure"] == "":
                    continue  # crashed trace, never scored
                key = (row["suite"], row["user_task"], row["injection_task"])
                sec[model][key].append(row["secure"] == "1")
                util[model][key].append(row["utility"] == "1")
            elif row["user_task"].startswith("injection_task"):
                goal[model][(row["suite"], row["user_task"])].append(row["utility"] == "1")
            else:
                clean[model][(row["suite"], row["user_task"])].append(row["utility"] == "1")

    def collapse(d):
        return {k: sum(v) * 2 > len(v) for k, v in d.items()}

    models = sorted(sec)
    # Distinct vote counts per model, so a caller can tell a completed run from
    # one whose second repetition is still in flight. See the guard in main().
    vote_shapes = {
        m: {len(v) for v in sec[m].values()} | {len(v) for v in goal[m].values()}
        for m in models
    }
    return (
        models,
        {m: collapse(sec[m]) for m in models},
        {m: collapse(util[m]) for m in models},
        {m: collapse(goal[m]) for m in models},
        {m: collapse(clean[m]) for m in models},
        degen,
        total,
        {m: len(reps[m]) for m in models},
        vote_shapes,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--max-reps",
        type=int,
        default=None,
        help=(
            "Keep only repetitions 1..N. Use this to level models carrying "
            "different repetition counts, since the majority threshold differs: "
            "two repetitions demand both, four demand three of four."
        ),
    )
    args = parser.parse_args()

    export_dir = args.export_dir
    if not (export_dir / "per_case.csv.gz").exists():
        print(f"No per_case.csv.gz under {export_dir}")
        return 1

    models, sec, util, goal, clean, degen, total, reps, vote_shapes = load(
        export_dir, max_reps=args.max_reps
    )
    manifest = {}
    if (export_dir / "manifest.json").exists():
        manifest = json.loads((export_dir / "manifest.json").read_text())

    out: dict = {"run_id": manifest.get("run_id"), "config_hash": manifest.get("config_hash")}

    print("=" * 92)
    print(f"ARM DASHBOARD   {export_dir.name}")
    if manifest:
        print(f"run {manifest.get('run_id')}   config_hash {manifest.get('config_hash')}   "
              f"attack {manifest.get('harness', {}).get('attack')}")
        print(f"measured cost ${manifest.get('total_cost_usd', 0.0):.2f}")
    print("=" * 92)

    n_goals = max((len(goal[m]) for m in models), default=0)
    n_clean = max((len(clean[m]) for m in models), default=0)
    # Anthropic ids run to 26 characters and OpenAI's to 13. A fixed width
    # silently swallows the column separator on the long ones, which is how a
    # table stops being readable exactly when somebody is skimming it for a
    # number.
    w = max((len(m) for m in models), default=8) + 2

    # A PARTIAL repetition silently corrupts every collapsed number in this
    # table, and it corrupts them downward. Majority collapse is
    # `sum(votes) * 2 > len(votes)`, so a case carrying two votes needs both,
    # while a case the second repetition has not reached yet carries one vote
    # and needs only that. Mixing the two understates the gate and the
    # utilities. Observed live: a gate read mid-run came out below the true
    # single-repetition value. Refuse to print quietly in that state.
    uneven = [m for m in models if len(vote_shapes[m]) > 1]
    if uneven:
        print("\n  *** PARTIAL REPETITION IN PROGRESS: " + ", ".join(uneven))
        print("  Cases below carry differing vote counts, so every collapsed figure is")
        print("  biased DOWNWARD. Read per-repetition numbers until the run completes.")

    # A model legitimately extended past its neighbours is not a partial run, and
    # the guard above will not see it: within that model every case has the same
    # vote count. But the majority THRESHOLD moves. At two repetitions
    # `sum(votes) * 2 > len(votes)` demands both, so the collapsed insecure set is
    # the UNION of failures; at four it demands three of four, which is far
    # stricter. Putting those two in one column understates the extended model
    # relative to the others and invents a difference the data does not contain.
    counts = {m: reps[m] for m in models}
    if len(set(counts.values())) > 1:
        spread = ", ".join(f"{m}={n}" for m, n in sorted(counts.items(), key=lambda kv: -kv[1]))
        common = min(counts.values())
        print(f"\n  *** MODELS CARRY DIFFERENT REPETITION COUNTS: {spread}")
        print("  The majority threshold is not the same at each: 2 repetitions demand")
        print("  both (a union of failures), 4 demand three of four. Any CROSS-MODEL")
        print("  row below mixes conventions and is not comparable.")
        print(f"  Re-run with --max-reps {common} for a comparable table, and read the")
        print("  extended model's own repetitions separately for its noise floor.")

    print("\nPer model")
    print(f"  {'model':<{w}}{'cases':>6}{'ASR':>8}{'95% CI':>18}"
          f"{'gate':>8}{'clean U':>9}{'U|attack':>10}{'degen':>7}{'reps':>6}")
    print("  " + "-" * (w + 74))
    rows = []
    for m in models:
        n = len(sec[m])
        hits = n - sum(sec[m].values())
        lo, hi = _wilson(hits, n)
        g = sum(goal[m].values())
        cu = sum(clean[m].values())
        ua = sum(util[m].values()) / n if n else 0.0
        dg = degen[m] / total[m] if total[m] else 0.0
        print(f"  {m:<{w}}{n:>6}{hits / n if n else 0:>7.2%}"
              f"{f'[{lo:.2%}, {hi:.2%}]':>18}{f'{g}/{n_goals}':>8}"
              f"{f'{cu}/{n_clean}':>9}{ua:>9.1%}{dg:>6.1%}{reps[m]:>6}")
        rows.append({"model": m, "cases": n, "attack_successes": hits,
                     "asr": hits / n if n else 0, "ci95": [lo, hi],
                     "goals_executable": g, "goals_total": n_goals,
                     "clean_utility": cu, "clean_total": n_clean,
                     "utility_under_attack": ua, "degenerate_share": dg,
                     "repetitions": reps[m]})
    out["per_model"] = rows

    print("\n  'gate' is injection goals the model executes when asked DIRECTLY, of")
    print(f"  {n_goals}. It is the load-bearing number when ASR is near zero: a low gate")
    print("  means the instrument has no range left on this model, not that the")
    print("  model resisted. 'clean U' is user tasks solved with no attack present.")

    print("\nPer suite, attack successes / cases")
    suites = sorted({k[0] for m in models for k in sec[m]})
    print(f"  {'model':<{w}}" + "".join(f"{s:>14}" for s in suites))
    print("  " + "-" * (w + 14 * len(suites)))
    per_suite = {}
    for m in models:
        cells = []
        per_suite[m] = {}
        for s in suites:
            keys = [k for k in sec[m] if k[0] == s]
            hits = sum(1 for k in keys if not sec[m][k])
            cells.append(f"{hits}/{len(keys)}")
            per_suite[m][s] = {"hits": hits, "cases": len(keys)}
        print(f"  {m:<{w}}" + "".join(f"{c:>14}" for c in cells))
    out["per_suite"] = per_suite

    print("\nAll ordered pairs. 'neg' is secure under A and insecure under B.")
    print("Same-generation pairs are TIER STEPS, not version transitions; the")
    print("arithmetic is identical and the meaning is not.")
    pw = 2 * w + 4
    print(f"  {'A -> B':<{pw}}{'gate':>9}{'cases':>7}{'neg':>5}{'pos':>5}"
          f"{'NFR':>8}{'McNemar':>10}")
    print("  " + "-" * (pw + 44))
    pairs = []
    for a, b in itertools.permutations(models, 2):
        shared_all = set(sec[a]) & set(sec[b])
        execu = {k for k, v in goal[a].items() if v} & {k for k, v in goal[b].items() if v}
        for label, shared in (("ungated", shared_all),
                              ("matched", {k for k in shared_all if (k[0], k[2]) in execu})):
            neg = sum(1 for k in shared if sec[a][k] and not sec[b][k])
            pos = sum(1 for k in shared if not sec[a][k] and sec[b][k])
            base = sum(1 for k in shared if sec[a][k])
            nfr = neg / base if base else 0.0
            p = _mcnemar_exact(neg, pos)
            print(f"  {a + ' -> ' + b:<{pw}}{label:>9}{len(shared):>7}{neg:>5}{pos:>5}"
                  f"{nfr:>7.2%}{p:>10.3f}")
            pairs.append({"before": a, "after": b, "gate": label,
                          "cases": len(shared), "negative_flips": neg,
                          "positive_flips": pos, "nfr": nfr, "mcnemar_p": p})
    out["pairs"] = pairs

    if args.json_out:
        args.json_out.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
