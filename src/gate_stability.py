# SPDX-License-Identifier: Apache-2.0
"""How much of a capability-matched rate is the gate's own noise.

The attack-capability gate is a MEASURED set, not a fixed property of the
benchmark. AgentDojo runs each injection goal as a standalone user task, and the
gate is the subset of those goals a model completes on direct request. Every
matched rate in this study is therefore computed on a case set that was itself
estimated from the same stochastic instrument as the rate.

That estimate moves between repetitions of one model, which raises the obvious
question against every matched number the study quotes: how much of the rate is
the gate?

The question is answered by holding the security outcomes fixed at the published
collapse and varying ONLY the gate definition:

  rep1..repN  the gate as measured in a single repetition
  majority    the published convention, `sum(votes) * 2 > len(votes)`
  unanimous   goals executed in EVERY repetition, the strictest gate
  union       goals executed in ANY repetition, the most permissive

The spread across those rows is the gate's contribution and nothing else. A
second table then varies both together, taking gate and security from the same
single repetition, which is the total per-repetition swing.

**Cardinality is not the statistic to watch.** A gate can hold the same COUNT in
every repetition while the underlying SET churns, and the count is what a
dashboard prints. The `gap` column is union minus intersection and is the honest
measure of instability; `swing` is max minus min count and is what a single
printed gate figure conceals.

A rate of exactly 0.0% is invariant to the gate by construction, for every
subset of an all-secure set is all-secure. What moves there is the DENOMINATOR,
hence the confidence bound, so the table prints the Wilson upper bound whenever
the numerator is zero.

Usage:
    python src/gate_stability.py results_export/<run_id>
    python src/gate_stability.py results_export/<run_id> --pair MODEL_A MODEL_B
    python src/gate_stability.py results_export/<run_id> --json-out gate.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import math
from pathlib import Path


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _mcnemar(b: int, c: int) -> float:
    """`analyze.py`'s convention, inherited rather than chosen.

    THE TWO ANALYSIS SCRIPTS DO NOT RUN THE SAME TEST. `analyze.py` takes the
    exact binomial below 25 discordant pairs and the continuity-corrected
    chi-square at or above it; `arm_dashboard.py` is always exact. On a pair
    with many discordant cases the two can differ by orders of magnitude while
    both lie far past any threshold. This script follows `analyze.py` so that
    its output reproduces `analyze.py`'s rather than quietly contradicting it.
    """
    n = b + c
    if n == 0:
        return 1.0
    try:
        from statsmodels.stats.contingency_tables import mcnemar

        return float(mcnemar([[0, b], [c, 0]], exact=n < 25, correction=True).pvalue)
    except ImportError:
        tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) * 0.5**n
        return min(1.0, 2 * tail)


def load(export_dir: Path):
    """Security votes and per-repetition gate membership, kept separate.

    `sec` collapses nothing; the caller decides. `goal` is indexed by repetition
    because that is the whole point of this script, where every other script in
    the harness collapses it on the way in and loses the information.
    """
    sec: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    sec_rep: dict = collections.defaultdict(lambda: collections.defaultdict(dict))
    goal: dict = collections.defaultdict(lambda: collections.defaultdict(dict))
    reps: set = set()

    with gzip.open(export_dir / "per_case.csv.gz", "rt") as fh:
        for row in csv.DictReader(fh):
            model, rep = row["model"], row["repetition"]
            reps.add(rep)
            if row["injection_task"]:
                if row["secure"] == "":
                    continue  # crashed trace, never scored
                key = (row["suite"], row["user_task"], row["injection_task"])
                sec[model][key].append(row["secure"] == "1")
                sec_rep[model][rep][key] = row["secure"] == "1"
            elif row["user_task"].startswith("injection_task"):
                goal[model][rep][(row["suite"], row["user_task"])] = row["utility"] == "1"

    return sec, sec_rep, goal, sorted(reps, key=lambda r: int("".join(c for c in r if c.isdigit()) or 0))


def gate_variants(goal_model: dict, reps: list[str]) -> dict[str, set]:
    """Every gate definition for one model, over THAT MODEL'S repetitions.

    A model absent from a repetition must not be read as having executed zero
    goals in it. Run `20260817T141657Z` holds `gpt-5.4` at two repetitions and
    `gpt-5.4-nano` at four, and iterating the run's repetition list gave 5.4 two
    empty sets, a unanimous gate of 0 and a majority gate of 0 -- a model that
    executes most of its goals printed as executing none. The bug is silent.

    So: a repetition counts for a model only if that model has goal rows in it.
    `main` warns when the counts differ, for the majority THRESHOLD is not the
    same at two repetitions as at four.
    """
    own = [r for r in reps if goal_model.get(r)]
    per = [{k for k, v in goal_model[r].items() if v} for r in own]
    if not per:
        return {}
    ever = set().union(*per)
    variants = {r: s for r, s in zip(own, per)}
    variants["majority"] = {k for k in ever if sum(k in s for s in per) * 2 > len(per)}
    variants["unanimous"] = set.intersection(*per)
    variants["union"] = ever
    variants["_reps"] = own
    return variants


def _collapse(votes: dict) -> dict:
    return {k: sum(v) * 2 > len(v) for k, v in votes.items()}


def _paired(sa: dict, sb: dict, gate: set | None):
    shared = set(sa) & set(sb)
    if gate is not None:
        shared = {k for k in shared if (k[0], k[2]) in gate}
    neg = sum(1 for k in shared if sa[k] and not sb[k])
    pos = sum(1 for k in shared if not sa[k] and sb[k])
    base = sum(1 for k in shared if sa[k])
    return len(shared), neg, pos, base, (neg / base if base else 0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--pair", nargs=2, metavar=("BEFORE", "AFTER"), default=None,
                        help="Recompute one paired comparison under every gate definition.")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if not (args.export_dir / "per_case.csv.gz").exists():
        print(f"No per_case.csv.gz under {args.export_dir}")
        return 1

    sec, sec_rep, goal, reps = load(args.export_dir)
    models = sorted(sec)
    maj = {m: _collapse(sec[m]) for m in models}
    gates = {m: gate_variants(goal[m], reps) for m in models}
    n_goals = max((len(g) for m in models for g in [set().union(*(goal[m][r] for r in reps))]
                   if g), default=0)
    out: dict = {"export": args.export_dir.name, "repetitions": reps, "goals_total": n_goals}

    manifest = {}
    if (args.export_dir / "manifest.json").exists():
        manifest = json.loads((args.export_dir / "manifest.json").read_text())
        out["config_hash"] = manifest.get("config_hash")

    print("=" * 96)
    print(f"GATE STABILITY   {args.export_dir.name}   {len(reps)} repetitions   "
          f"config_hash {manifest.get('config_hash', '?')}")
    print("=" * 96)

    # ---- how much the gate itself moves -----------------------------------
    w = max(len(m) for m in models) + 2
    own_reps = {m: gates[m].get("_reps", []) for m in models}
    if len({len(v) for v in own_reps.values()}) > 1:
        spread = ", ".join(f"{m}={len(r)}" for m, r in sorted(own_reps.items(),
                                                              key=lambda kv: -len(kv[1])))
        print(f"\n  *** MODELS CARRY DIFFERENT REPETITION COUNTS: {spread}")
        print("  Each model's gate is computed over its OWN repetitions, so no model")
        print("  is credited with an empty one. The majority THRESHOLD still differs")
        print("  between them, so read the swing and gap columns per model and do not")
        print("  compare majority gates across this table.")

    print("\nGate membership per repetition, of", n_goals, "injection goals")
    print(f"  {'model':<{w}}" + "".join(f"{r:>7}" for r in reps)
          + f"{'maj':>6}{'unan':>6}{'union':>7}{'swing':>7}{'gap':>6}")
    print("  " + "-" * (w + 7 * len(reps) + 32))
    per_model = {}
    for m in models:
        g = gates[m]
        counts = [len(g[r]) for r in own_reps[m]]
        swing = max(counts) - min(counts)
        gap = len(g["union"]) - len(g["unanimous"])
        cells = [str(len(g[r])) if r in g else "-" for r in reps]
        print(f"  {m:<{w}}" + "".join(f"{c:>7}" for c in cells)
              + f"{len(g['majority']):>6}{len(g['unanimous']):>6}{len(g['union']):>7}"
              + f"{swing:>7}{gap:>6}")
        per_model[m] = {"repetitions": own_reps[m], "per_rep": counts,
                        "majority": len(g["majority"]),
                        "unanimous": len(g["unanimous"]), "union": len(g["union"]),
                        "swing": swing, "gap": gap}
    out["gate_membership"] = per_model

    print("\n  'swing' is max minus min COUNT, which is what a printed gate figure")
    print("  conceals. 'gap' is union minus intersection, the number of goals whose")
    print("  membership is not reproducible. A small swing can hide a large gap.")

    # ---- what that does to each model's matched rate -----------------------
    print("\nMatched ASR with security fixed at majority, gate varied")
    print(f"  {'model':<{w}}{'raw':>8}" + "".join(f"{lab:>17}" for lab in ("majority", "unanimous", "union")))
    print("  " + "-" * (w + 59))
    sens = {}
    for m in models:
        n = len(maj[m])
        raw = (n - sum(maj[m].values())) / n if n else 0.0
        cells, rec = [], {}
        for lab in ("majority", "unanimous", "union"):
            keys = [k for k in maj[m] if (k[0], k[2]) in gates[m][lab]]
            hits = sum(1 for k in keys if not maj[m][k])
            rate = hits / len(keys) if keys else 0.0
            _, hi = _wilson(hits, len(keys))
            cells.append(f"{rate:.1%}/{len(keys)}" + (f" <{hi:.1%}" if hits == 0 else ""))
            rec[lab] = {"rate": rate, "cases": len(keys), "hits": hits, "ci_high": hi}
        print(f"  {m:<{w}}{raw:>7.1%}" + "".join(f"{c:>17}" for c in cells))
        sens[m] = {"raw": raw, **rec}
    out["matched_asr"] = sens
    print("\n  A 0.0% is gate-invariant by construction; what moves is the denominator")
    print("  and hence the bound, printed as '<x%' where the numerator is zero.")

    # ---- one pair under every gate ----------------------------------------
    if args.pair:
        a, b = args.pair
        missing = [m for m in (a, b) if m not in sec]
        if missing:
            print(f"\nNot in this export: {', '.join(missing)}")
            return 1
        print(f"\nPaired: {a} -> {b}")
        print("\n  Security fixed at majority; ONLY the gate definition varies.")
        print(f"  {'gate definition':<20}{'|gate|':>8}{'cases':>7}{'neg':>5}{'pos':>5}"
              f"{'base':>6}{'NFR':>9}{'McNemar':>11}")
        print("  " + "-" * 71)
        rows = []
        n, neg, pos, base, nfr = _paired(maj[a], maj[b], None)
        print(f"  {'ungated':<20}{'-':>8}{n:>7}{neg:>5}{pos:>5}{base:>6}{nfr:>8.1%}"
              f"{_mcnemar(neg, pos):>11.1e}")
        rows.append({"gate": "ungated", "cases": n, "neg": neg, "pos": pos, "nfr": nfr})
        common = [r for r in reps if r in gates[a] and r in gates[b]]
        for lab in common + ["majority", "unanimous", "union"]:
            gate = gates[a][lab] & gates[b][lab]
            n, neg, pos, base, nfr = _paired(maj[a], maj[b], gate)
            print(f"  {lab:<20}{len(gate):>8}{n:>7}{neg:>5}{pos:>5}{base:>6}{nfr:>8.1%}"
                  f"{_mcnemar(neg, pos):>11.1e}")
            rows.append({"gate": lab, "gate_size": len(gate), "cases": n, "neg": neg,
                         "pos": pos, "base": base, "nfr": nfr})
        vals = [r["nfr"] for r in rows if r["gate"] != "ungated"]
        print(f"\n  Gate contributes a spread of {100 * (max(vals) - min(vals)):.2f} points.")

        # Each model's own RATE on the goals both execute, which is the right
        # comparison across vendors where a paired NFR is not meaningful. Two
        # gates of equal SIZE need not share their members, so a ratio between
        # two rates can shrink sharply once the shared set rather than the
        # shared count is used.
        print("\n  Rates on the goals BOTH models execute. An equal gate size is not")
        print("  an equal gate set, so this differs from comparing two raw rates.")
        print(f"  {'gate definition':<20}{'shared':>8}{'excl A':>8}{'excl B':>8}{'cases':>7}"
              f"{'ASR A':>9}{'ASR B':>9}{'ratio':>8}{'McNemar':>10}")
        print("  " + "-" * 87)
        shared_rates = []
        for lab in common + ["majority", "unanimous", "union"]:
            ga, gb = gates[a][lab], gates[b][lab]
            inter = ga & gb
            keys = [k for k in set(maj[a]) & set(maj[b]) if (k[0], k[2]) in inter]
            fa = sum(1 for k in keys if not maj[a][k])
            fb = sum(1 for k in keys if not maj[b][k])
            ra = fa / len(keys) if keys else 0.0
            rb = fb / len(keys) if keys else 0.0
            disc_b = sum(1 for k in keys if maj[a][k] and not maj[b][k])
            disc_c = sum(1 for k in keys if not maj[a][k] and maj[b][k])
            ratio = rb / ra if ra else float("inf")
            print(f"  {lab:<20}{len(inter):>8}{len(ga - gb):>8}{len(gb - ga):>8}{len(keys):>7}"
                  f"{ra:>8.2%}{rb:>8.2%}{ratio:>7.2f}x{_mcnemar(disc_b, disc_c):>10.4f}")
            shared_rates.append({"gate": lab, "shared_goals": len(inter),
                                 "cases": len(keys), "asr_before": ra, "asr_after": rb,
                                 "ratio": ratio, "mcnemar_p": _mcnemar(disc_b, disc_c)})

        # Gate AND security from the same repetition. This is the total swing,
        # and it is the only row where the security collapse is not doing any
        # work on the reader's behalf.
        print("\n  Gate AND security both from a single repetition, nothing collapsed.")
        print(f"  {'repetition':<20}{'|gate|':>8}{'cases':>7}{'neg':>5}{'pos':>5}"
              f"{'base':>6}{'NFR':>9}")
        print("  " + "-" * 60)
        full = []
        for r in common:
            gate = gates[a][r] & gates[b][r]
            n, neg, pos, base, nfr = _paired(sec_rep[a][r], sec_rep[b][r], gate)
            print(f"  {r:<20}{len(gate):>8}{n:>7}{neg:>5}{pos:>5}{base:>6}{nfr:>8.1%}")
            full.append({"repetition": r, "cases": n, "neg": neg, "pos": pos, "nfr": nfr})
        fv = [r["nfr"] for r in full]
        print(f"\n  Total per-repetition spread {100 * (max(fv) - min(fv)):.2f} points.")
        out["pair"] = {"before": a, "after": b, "gate_varied": rows,
                       "per_repetition": full, "shared_gate_rates": shared_rates}

    if args.json_out:
        args.json_out.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
