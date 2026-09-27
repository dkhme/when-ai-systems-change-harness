# SPDX-License-Identifier: Apache-2.0
"""Pool several result exports into one, refusing to pool across configurations.

An arm does not always arrive in a single sweep. The GPT arm was measured in two:
four models on 2026-08-17 morning, then the two gpt-5.4 predecessors that
afternoon as Episode B. Every cross-lineage figure needs them in one table.

One invariant makes such a table meaningful: if one model ran under one
configuration and another under a different one, the comparison is void and the
money is wasted. Pooling by hand
with `cat` cannot check that. This can, and refuses.

    python src/pool_exports.py results_export/<a> results_export/<b> --out results_export/<name>

The pooled directory carries `per_case.csv.gz` in the same schema, so
`arm_dashboard.py` and `flip_anatomy.py` read it unchanged, plus a
`pooled_manifest.json` recording exactly which runs went in.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

# Fields that must agree for two runs to belong to one configuration generation.
# config_hash already covers harness, generation, attack and suites; the rest are
# listed so a mismatch names the offending field rather than only the hash.
HARNESS_KEYS = ["agentdojo_version", "benchmark_version", "attack", "defense", "tool_output_format"]
GENERATION_KEYS = ["temperature", "thinking_mode", "max_output_tokens"]


def load_manifest(d: Path) -> dict:
    p = d / "manifest.json"
    if not p.exists():
        raise SystemExit(f"{d} has no manifest.json. A directory with no manifest is not a run.")
    return json.loads(p.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("exports", nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--allow-config-mismatch",
        action="store_true",
        help=(
            "Pool anyway. The result is NOT a valid cross-model comparison and the "
            "pooled manifest records that it was forced."
        ),
    )
    args = ap.parse_args()

    manifests = {d.name: load_manifest(d) for d in args.exports}

    # ---- the invariant -----------------------------------------------------
    problems: list[str] = []
    ref_name, ref = next(iter(manifests.items()))
    for name, m in manifests.items():
        if m["config_hash"] != ref["config_hash"]:
            problems.append(
                f"config_hash differs: {name}={m['config_hash']} vs {ref_name}={ref['config_hash']}"
            )
        for k in HARNESS_KEYS:
            if m["harness"].get(k) != ref["harness"].get(k):
                problems.append(
                    f"harness.{k} differs: {name}={m['harness'].get(k)!r} vs "
                    f"{ref_name}={ref['harness'].get(k)!r}"
                )
        for k in GENERATION_KEYS:
            if m["generation"].get(k) != ref["generation"].get(k):
                problems.append(
                    f"generation.{k} differs: {name}={m['generation'].get(k)!r} vs "
                    f"{ref_name}={ref['generation'].get(k)!r}"
                )
        if sorted(m.get("suites") or []) != sorted(ref.get("suites") or []):
            problems.append(f"suites differ: {name}={m.get('suites')} vs {ref_name}={ref.get('suites')}")

    if problems:
        print("THESE RUNS DO NOT BELONG TO ONE CONFIGURATION GENERATION:")
        for p in dict.fromkeys(problems):
            print(f"  - {p}")
        if not args.allow_config_mismatch:
            print(
                "\nRefusing to pool. A cross-model comparison across configurations is "
                "void. Re-run the odd one "
                "out, or pass --allow-config-mismatch and label every figure as invalid."
            )
            return 1
        print("\n--allow-config-mismatch given. Pooling anyway. The output is not citable.")

    # ---- pool --------------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    header: list[str] | None = None
    counts: dict[str, int] = {}
    models: dict[str, str] = {}
    seen: set[tuple] = set()
    duplicates = 0

    with gzip.open(args.out / "per_case.csv.gz", "wt", newline="") as fh:
        writer = csv.writer(fh)
        for d in args.exports:
            with gzip.open(d / "per_case.csv.gz", "rt") as src:
                reader = csv.reader(src)
                cols = next(reader)
                if header is None:
                    header = cols
                    writer.writerow(header)
                elif cols != header:
                    raise SystemExit(f"{d} has a different per_case schema: {cols}")
                idx = {c: i for i, c in enumerate(header)}
                n = 0
                for row in reader:
                    # A model measured in two runs would silently double its
                    # repetition count and corrupt every collapse. Drop exact
                    # (model, rep, case) repeats and report how many.
                    key = (
                        row[idx["model"]],
                        row[idx["repetition"]],
                        row[idx["suite"]],
                        row[idx["user_task"]],
                        row[idx["injection_task"]],
                    )
                    if key in seen:
                        duplicates += 1
                        continue
                    seen.add(key)
                    writer.writerow(row)
                    n += 1
                    models.setdefault(row[idx["model"]], d.name)
                counts[d.name] = n

    pooled = {
        "pooled_from": [
            {
                "run_id": m.get("run_id"),
                "export": name,
                "config_hash": m["config_hash"],
                "models": m.get("models"),
                "cost_usd": m.get("total_cost_usd"),
                "rows_taken": counts.get(name),
            }
            for name, m in manifests.items()
        ],
        "config_hash": ref["config_hash"],
        "harness": ref["harness"],
        "generation": ref["generation"],
        "suites": ref.get("suites"),
        "model_source": models,
        "duplicate_rows_dropped": duplicates,
        "config_mismatch_forced": bool(problems),
        # Deliberately NOT summed into a single cost figure. Each source manifest
        # understates its own run when repetitions resumed from cache, so a sum of
        # manifest fields is not the true spend.
        "cost_note": (
            "Per-run cost_usd above is the manifest field, which reads 0.0000 for any "
            "repetition served from cache. Do not sum these as the true spend."
        ),
    }
    (args.out / "pooled_manifest.json").write_text(json.dumps(pooled, indent=2) + "\n")
    # arm_dashboard.py looks for manifest.json; give it one that is honest about
    # being a pool rather than a run.
    (args.out / "manifest.json").write_text(json.dumps(pooled | {"run_id": args.out.name}, indent=2) + "\n")

    print(f"pooled {sum(counts.values()):,} rows from {len(args.exports)} exports -> {args.out}")
    for name, n in counts.items():
        print(f"  {name}: {n:,} rows")
    if duplicates:
        print(f"  dropped {duplicates:,} duplicate (model, repetition, case) rows")
    print(f"  models: {', '.join(sorted(models))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
