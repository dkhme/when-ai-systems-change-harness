# SPDX-License-Identifier: Apache-2.0
"""Export a compact, publishable dataset from a sweep's raw traces.

A completed sweep writes roughly 300 MB of trace files, most of it the full
message history of every trajectory. That is worth keeping locally, for it is
the evidence behind any disputed case, but it does not belong in the repository.

This produces the small artifact that does: one row per (model, repetition,
case) carrying the scored outcome, which is everything needed to recompute the
attack success rates, the negative-flip rates, the McNemar tests, and the
confidence intervals. The raw traces stay on disk and can be re-exported at any
time.

Written to results_export/<run_id>/:

  per_case.csv.gz      one row per model, repetition, and case
  summary.md           the analysis as rendered for a reader
  comparisons.json     the paired version-to-version statistics
  manifest.json        the run's configuration, cost, and token totals
  models.csv           per-model aggregates
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import analyze  # noqa: E402
import costs  # noqa: E402

EXPORT_ROOT = SRC.parent / "results_export"
CONFIG_DIR = SRC.parent / "config"


def export(run_dir: Path) -> Path:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    run_id = manifest.get("run_id", run_dir.name)
    attack = (manifest.get("harness") or {}).get("attack")

    out_dir = EXPORT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # A model is addressed by the DIRECTORY its traces live under, which is the
    # pipeline name. Arms whose served ids contain a path separator alias them
    # and record the mapping; manifests without it are pre-alias Google runs
    # where the served id already is the directory name.
    aliases: dict[str, str] = manifest.get("model_aliases") or {}
    alias_to_id = {alias: served for served, alias in aliases.items()}

    # THE DISK IS THE GROUND TRUTH FOR WHAT WAS RUN, NOT THE MANIFEST.
    #
    # Reading the model list from the manifest alone silently discards measured
    # data, and it did. Resuming a run with `--models <subset>` rewrites the
    # manifest to name only that subset, because the manifest describes the
    # INVOCATION rather than the run directory. Episode D-prime resumed
    # 20260817T141657Z with `--models gpt-5.4-nano` to add two repetitions, and
    # the next export dropped gpt-5.4's 934 completed traces without a word:
    # they were on disk the whole time and simply were not asked for.
    #
    # A dropped model is worse than a crash here. The export is the committed
    # artifact from which every figure is recomputed, so the loss propagates
    # into the pooled table and the write-up while every file still looks
    # well-formed.
    #
    # So: union the manifest's list with every directory that actually holds
    # repetitions, and keep only those with traces. The manifest still supplies
    # the alias mapping, which the directory scan cannot recover.
    on_disk = {d.name for d in run_dir.iterdir() if d.is_dir() and any(d.glob("rep*"))}
    declared = set(aliases.values())
    models = sorted(declared | on_disk)
    # A manifest names every model the sweep INTENDS to run, so exporting a run
    # that is still in flight, or one stopped at the budget ceiling, would break
    # on the first model with no directory yet. Export what exists.
    models = [m for m in models if m in on_disk]
    if declared - on_disk:
        print(f"  note: declared but not yet run, skipped: {', '.join(sorted(declared - on_disk))}")
    if on_disk - declared:
        print(
            f"  note: on disk but absent from the manifest, exported anyway: "
            f"{', '.join(sorted(on_disk - declared))}"
        )

    # ---- per-case rows -----------------------------------------------------
    rows = 0
    with gzip.open(out_dir / "per_case.csv.gz", "wt", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "model",
                "repetition",
                "suite",
                "user_task",
                "injection_task",
                "attack",
                "utility",
                "secure",
                "degenerate",
                "duration_s",
            ]
        )
        for model in models:
            for rep_dir in sorted(
                d for d in (run_dir / model).iterdir() if d.is_dir() and d.name.startswith("rep")
            ):
                rep = rep_dir.name
                for path in rep_dir.rglob("*.json"):
                    if path.name == "manifest.json":
                        continue
                    try:
                        data = json.loads(path.read_text())
                    except (json.JSONDecodeError, OSError):
                        continue
                    if not isinstance(data, dict) or "suite_name" not in data:
                        continue
                    if not isinstance(data.get("utility"), bool):
                        continue
                    injection = data.get("injection_task_id")
                    if injection is not None and not isinstance(data.get("security"), bool):
                        continue
                    if attack and injection is not None and data.get("attack_type") != attack:
                        continue
                    writer.writerow(
                        [
                            model,
                            rep,
                            data["suite_name"],
                            data["user_task_id"],
                            injection or "",
                            data.get("attack_type") or "",
                            int(data["utility"]),
                            # Inverted from AgentDojo's `security`, which is True
                            # when the ATTACK SUCCEEDED. See analyze.py.
                            "" if injection is None else int(not data["security"]),
                            int(analyze._is_degenerate(data)),
                            round(float(data.get("duration", 0) or 0), 2),
                        ]
                    )
                    rows += 1

    # ---- per-model aggregates ---------------------------------------------
    pricing = costs.load_pricing(
        CONFIG_DIR / manifest.get("model_registry", "models_google.yaml")
    )
    outcomes = {}
    for model in models:
        try:
            outcomes[model] = analyze.load_model(run_dir, model, "majority", attack)
        except FileNotFoundError:
            continue

    with (out_dir / "models.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "model",
                "lineage",
                "lineage_order",
                "input_usd_per_mtok",
                "cases",
                "attack_success_rate",
                "asr_ci_low",
                "asr_ci_high",
                "repetitions",
                "cases_split_across_reps",
                "degenerate_share",
                "excluded_traces",
            ]
        )
        for model, out in outcomes.items():
            total = len(out.secure)
            insecure = total - sum(out.secure.values())
            asr = insecure / total if total else 0.0
            low, high = analyze._confint(insecure, total)
            spec = pricing.get(alias_to_id.get(model, model))
            writer.writerow(
                [
                    model,
                    spec.lineage if spec else "",
                    spec.order if spec else "",
                    spec.input if spec else "",
                    total,
                    round(asr, 6),
                    round(low, 6),
                    round(high, 6),
                    out.repetitions,
                    out.split_security,
                    round(out.degenerate_tasks / out.total_tasks, 6) if out.total_tasks else 0,
                    out.unusable_tasks,
                ]
            )

    # ---- paired comparisons -------------------------------------------------
    lineages: dict[str, list[str]] = {}
    for model in outcomes:
        served = alias_to_id.get(model, model)
        if served in pricing:
            lineages.setdefault(pricing[served].lineage, []).append(model)
    for chain in lineages.values():
        chain.sort(key=lambda m: pricing[alias_to_id.get(m, m)].order)

    comparisons = []
    for lineage, chain in lineages.items():
        for before_id, after_id in zip(chain, chain[1:]):
            result = analyze.compare(outcomes[before_id], outcomes[after_id])
            result["lineage"] = lineage
            comparisons.append(result)
    (out_dir / "comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")

    # ---- rendered summary and manifest -------------------------------------
    rendered = subprocess.run(
        [sys.executable, str(SRC / "analyze.py"), str(run_dir)],
        capture_output=True,
        text=True,
    )
    (out_dir / "summary.md").write_text(
        f"# AgentDojo across Gemini versions: run {run_id}\n\n"
        "Generated by `src/export_results.py`. Raw trajectories are kept locally\n"
        "and are not committed; `per_case.csv.gz` carries every scored outcome,\n"
        "which is sufficient to recompute every figure below.\n\n"
        "```\n" + rendered.stdout + "```\n"
    )
    if manifest:
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"exported {rows:,} case rows for {len(models)} models -> {out_dir}")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else Path.cwd() / args.run_dir
    if not run_dir.exists():
        print(f"No such run directory: {run_dir}")
        return 1
    export(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
