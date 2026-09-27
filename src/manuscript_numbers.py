# SPDX-License-Identifier: Apache-2.0
"""Every number the experimental section quotes, recomputed from the exports.

A number quoted in a write-up is not automatically the number the committed
export holds. Mixing conventions produces divergences that are not arithmetic
errors: single-repetition utility printed beside collapsed attack success rates
in one table, or utility over user tasks alone presented as clean utility over
every utility run.

So the manuscript takes its numbers from here rather than from the write-ups.
This script owns one convention and states it everywhere: **majority collapse
across repetitions**, `sum(votes) * 2 > len(votes)`, which at two repetitions is
the union of failures and at three demands two of three. Where a figure is
convention-sensitive the script prints the alternatives instead of choosing.

    python src/manuscript_numbers.py                 # every arm
    python src/manuscript_numbers.py --arm gemini    # one arm
    python src/manuscript_numbers.py > NUMBERS.txt

Companion scripts, each owning one question: `gate_stability.py` for how much of
a matched rate is the gate, `utility_decomposition.py` for U against G,
`noise_floor.py` for a model against itself, `arm_dashboard.py` for every ordered
pair, `analyze.py` for the lineage view.
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

from gate_stability import _mcnemar, _wilson, gate_variants  # noqa: E402

EXPORTS = SRC.parent / "results_export"

# The collapse this file owns. Stated once so no table can drift from it.
GATE_DEF = "majority"

# An arm is one export plus the lineages the paper reads out of it. Lineage
# order is stated here rather than read from the registry because the registry
# files an id under exactly one lineage, and several models sit in two
# structures: gpt-5.6-sol is the flagship successor and the top of the 5.6 tier
# ladder at once. `max_reps` levels a model extended past its neighbours, since
# the majority threshold is not the same at two repetitions as at four.
ARMS = {
    "gemini": {
        "export": "20260807T044342Z",
        "label": "Gemini, important_instructions, 4 suites, 3 repetitions",
        "lineages": {
            "flash-lite": ["gemini-2.5-flash-lite", "gemini-3.1-flash-lite",
                           "gemini-3.5-flash-lite"],
            "flash": ["gemini-2.5-flash", "gemini-3-flash-preview", "gemini-3.5-flash",
                      "gemini-3.6-flash"],
            "pro": ["gemini-2.5-pro", "gemini-3.1-pro-preview"],
        },
    },
    "anthropic": {
        "export": "20260810T044952Z",
        "label": "Anthropic Sonnet, important_instructions_no_names, 3 suites, 2 repetitions",
        "lineages": {"sonnet": ["claude-sonnet-4-5-20250929", "claude-sonnet-4-6",
                                "claude-sonnet-5"]},
    },
    "gpt": {
        "export": "gpt_pooled",
        "label": "GPT, important_instructions, 3 suites, 2 repetitions (levelled)",
        "max_reps": 2,
        "lineages": {
            "flagship": ["gpt-5.5", "gpt-5.6-sol"],
            "mid": ["gpt-5.4", "gpt-5.6-terra"],
            "nano": ["gpt-5.4-nano", "gpt-5.6-luna"],
            "5.6 tier ladder (NOT a version transition)": ["gpt-5.6-luna", "gpt-5.6-terra",
                                                           "gpt-5.6-sol"],
        },
    },
    "openweight": {
        "export": "20260809T020755Z",
        "label": "Open-weight via Baseten, important_instructions_no_names, 3 suites, 2 reps",
        "lineages": {"kimi": ["moonshotai__Kimi-K2.6", "moonshotai__Kimi-K2.7-Code",
                              "moonshotai__Kimi-K3"],
                     "glm": ["zai-org__GLM-4.7", "zai-org__GLM-5.2"]},
    },
    "crossvendor": {
        "export": "crossvendor_nonames",
        "label": "Cross-vendor pool, important_instructions_no_names, 3 suites, 2 reps",
        "lineages": {},
        "cross_pairs": [("claude-sonnet-5", "gpt-5.4-nano"),
                        ("claude-sonnet-5", "gpt-5.6-luna")],
    },
}


def _rep_index(label: str) -> int:
    digits = "".join(c for c in label if c.isdigit())
    return int(digits) if digits else 0


def load(export_dir: Path, max_reps: int | None = None):
    sec: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    sec_rep: dict = collections.defaultdict(lambda: collections.defaultdict(dict))
    goal: dict = collections.defaultdict(lambda: collections.defaultdict(dict))
    u: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    ua: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    degen: dict = collections.Counter()
    total: dict = collections.Counter()
    reps: dict = collections.defaultdict(set)

    with gzip.open(export_dir / "per_case.csv.gz", "rt") as fh:
        for row in csv.DictReader(fh):
            model, rep = row["model"], row["repetition"]
            if max_reps is not None and _rep_index(rep) > max_reps:
                continue
            reps[model].add(rep)
            total[model] += 1
            degen[model] += int(row["degenerate"] or 0)
            key = (row["suite"], row["user_task"])
            if row["injection_task"]:
                if row["secure"] == "":
                    continue
                k = key + (row["injection_task"],)
                sec[model][k].append(row["secure"] == "1")
                sec_rep[model][rep][k] = row["secure"] == "1"
                ua[model][k].append(row["utility"] == "1")
            elif row["user_task"].startswith("injection_task"):
                goal[model][rep][key] = row["utility"] == "1"
            else:
                u[model][key].append(row["utility"] == "1")
    return sec, sec_rep, goal, u, ua, degen, total, reps


def collapse(d):
    return {k: sum(v) * 2 > len(v) for k, v in d.items()}


def floors(sec_rep: dict) -> tuple[int, int, int]:
    """Directional noise floor over every ordered repetition pair.

    A one-pair floor is a point estimate, and the floor of a single model can
    span a wide range across ordered pairs, so the range matters more than the
    value. Returns (min, max, n_pairs); n_pairs of 1 is the warning.
    """
    labels = sorted(sec_rep, key=_rep_index)
    vals = []
    for a in labels:
        for b in labels:
            if a == b:
                continue
            shared = set(sec_rep[a]) & set(sec_rep[b])
            vals.append(sum(1 for k in shared if sec_rep[a][k] and not sec_rep[b][k]))
    if not vals:
        return (0, 0, 0)
    return (min(vals), max(vals), len(vals))


def report_arm(name: str, spec: dict) -> None:
    export = EXPORTS / spec["export"]
    if not (export / "per_case.csv.gz").exists():
        print(f"\n### {name}: NO EXPORT at {export}. Nothing here is reproducible.\n")
        return
    sec, sec_rep, goal, u, ua, degen, total, reps = load(export, spec.get("max_reps"))
    manifest = {}
    if (export / "manifest.json").exists():
        manifest = json.loads((export / "manifest.json").read_text())
    models = sorted(sec)
    maj = {m: collapse(sec[m]) for m in models}
    gates = {m: gate_variants(goal[m], sorted(reps[m], key=_rep_index)) for m in models}
    all_goals = {k for m in models for r in goal[m] for k in goal[m][r]}
    n_goals = len(all_goals)
    n_u = max((len(u[m]) for m in models), default=0)

    print("=" * 100)
    print(f"ARM: {name}   {spec['label']}")
    print(f"  export {spec['export']}   config_hash {manifest.get('config_hash')}   "
          f"attack {(manifest.get('harness') or {}).get('attack')}")
    # The manifest's cost field is written per invocation, so on a resumed run it
    # records the increment rather than the arm. Printed with that warning rather
    # than dropped, because a reader who finds it in the JSON deserves to know why
    # it is small.
    if manifest.get("total_cost_usd"):
        print(f"  manifest cost ${manifest['total_cost_usd']:.2f} "
              f"(per-invocation, NOT the arm total)")
    print("=" * 100)

    w = max(len(m) for m in models) + 2
    print(f"\n{'model':<{w}}{'cases':>7}{'fails':>7}{'ASR':>8}{'95% CI':>18}"
          f"{'gate':>8}{'floor':>10}{'U':>11}{'G':>9}{'UA':>8}{'degen':>7}{'reps':>6}")
    print("-" * (w + 106))
    for m in models:
        n = len(maj[m])
        hits = n - sum(maj[m].values())
        lo, hi = _wilson(hits, n)
        lo_f, hi_f, npair = floors(sec_rep[m])
        cu, cg = collapse(u[m]), collapse({k: v for k, v in goal[m].items()} and
                                          {k: [goal[m][r][k] for r in sorted(reps[m], key=_rep_index)
                                               if k in goal[m][r]]
                                           for k in gates[m].get("union", ())})
        su, sg = sum(cu.values()), sum(cg.values())
        cua = collapse(ua[m])
        floor_s = f"{lo_f}" if lo_f == hi_f else f"{lo_f}-{hi_f}"
        print(f"{m:<{w}}{n:>7}{hits:>7}{hits / n if n else 0:>7.2%}"
              f"{f'[{lo:.2%}, {hi:.2%}]':>18}"
              f"{f'{len(gates[m].get(GATE_DEF, ()))}/{n_goals}':>8}"
              f"{floor_s + f'({npair}p)':>10}"
              f"{f'{su}/{len(cu)}':>11}{f'{sg}/{n_goals}':>9}"
              f"{sum(cua.values()) / len(cua) if cua else 0:>7.1%}"
              f"{degen[m] / total[m] if total[m] else 0:>6.1%}{len(reps[m]):>6}")
    print(f"\n  U is over {n_u} user tasks, G over {n_goals} injection goals run standalone.")
    print("  clean utility is (U + G) / (n_u + n_goals) and is NOT a capability measure.")
    print("  floor is the directional noise floor over all ordered repetition pairs;")
    print("  '(1p)' marks a single-pair point estimate, which can move substantially")
    print("  from one repetition pair to another.")

    for lineage, chain in spec.get("lineages", {}).items():
        chain = [m for m in chain if m in maj]
        if len(chain) < 2:
            continue
        print(f"\nLINEAGE {lineage}: {' -> '.join(chain)}")
        tw = max(len(f"{a} -> {b}  [matched]") for a in chain for b in chain) + 2
        print(f"  {'transition':<{tw}}{'gate':>6}{'cases':>7}{'neg':>5}{'pos':>5}"
              f"{'NFR':>8}{'McNemar':>11}{'floor':>8}")
        print("  " + "-" * (tw + 45))
        pairs = list(zip(chain, chain[1:]))
        if len(chain) > 2:
            pairs.append((chain[0], chain[-1]))   # the ENDPOINT, see below
        for a, b in pairs:
            for label in ("raw", "matched"):
                shared = set(maj[a]) & set(maj[b])
                gate = gates[a]["majority"] & gates[b]["majority"]
                if label == "matched":
                    shared = {k for k in shared if (k[0], k[2]) in gate}
                neg = sum(1 for k in shared if maj[a][k] and not maj[b][k])
                pos = sum(1 for k in shared if not maj[a][k] and maj[b][k])
                base = sum(1 for k in shared if maj[a][k])
                lo_f, hi_f, _ = floors(sec_rep[a])
                print(f"  {f'{a} -> {b}  [{label}]':<{tw}}"
                      f"{len(gate) if label == 'matched' else n_goals:>6}{len(shared):>7}"
                      f"{neg:>5}{pos:>5}{neg / base if base else 0:>7.1%}"
                      f"{_mcnemar(neg, pos):>11.2e}"
                      f"{(str(lo_f) if lo_f == hi_f else f'{lo_f}-{hi_f}'):>8}"
                      + ("   <- ENDPOINT" if (a, b) == (chain[0], chain[-1])
                         and len(chain) > 2 and label == "matched" else ""))
        if len(chain) > 2:
            print("  The ENDPOINT row is not the sum of the consecutive rows, for the gate")
            print("  chain is NOT transitive: gate(A,B) & gate(B,C) != gate(A,C). Never quote")
            print("  a gated comparison without the raw row beside it.")

    for a, b in spec.get("cross_pairs", []):
        if a not in maj or b not in maj:
            continue
        print(f"\nCROSS-VENDOR RATES on goals BOTH execute: {a} vs {b}")
        print(f"  {'gate def':<14}{'shared':>8}{'excl A':>8}{'excl B':>8}{'cases':>7}"
              f"{'ASR A':>9}{'ASR B':>9}{'ratio':>8}{'McNemar':>10}")
        print("  " + "-" * 81)
        own = sorted(set(reps[a]) & set(reps[b]), key=_rep_index)
        for d in own + ["majority", "unanimous", "union"]:
            if d not in gates[a] or d not in gates[b]:
                continue
            inter = gates[a][d] & gates[b][d]
            keys = [k for k in set(maj[a]) & set(maj[b]) if (k[0], k[2]) in inter]
            fa = sum(1 for k in keys if not maj[a][k])
            fb = sum(1 for k in keys if not maj[b][k])
            ra = fa / len(keys) if keys else 0.0
            rb = fb / len(keys) if keys else 0.0
            db = sum(1 for k in keys if maj[a][k] and not maj[b][k])
            dc = sum(1 for k in keys if not maj[a][k] and maj[b][k])
            print(f"  {d:<14}{len(inter):>8}{len(gates[a][d] - gates[b][d]):>8}"
                  f"{len(gates[b][d] - gates[a][d]):>8}{len(keys):>7}{ra:>8.2%}{rb:>8.2%}"
                  f"{(rb / ra if ra else float('inf')):>7.2f}x{_mcnemar(db, dc):>10.4f}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arm", choices=sorted(ARMS), default=None)
    args = parser.parse_args()
    print("MANUSCRIPT NUMBERS. Majority collapse throughout unless a table says")
    print("otherwise. Generated by src/manuscript_numbers.py; do not hand-edit.")
    print()
    for name in ([args.arm] if args.arm else list(ARMS)):
        report_arm(name, ARMS[name])
    print("=" * 100)
    print("OPUS LINEAGE: no export exists.")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
