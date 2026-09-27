# SPDX-License-Identifier: Apache-2.0
"""Run the AgentDojo sweep across Gemini versions.

The released AgentDojo runner is not usable for this study as it stands. Four
properties are missing, and each is supplied here.

Concurrency. `--max-workers` parallelises across suites only, so at most four
tasks are ever in flight and the run is bounded by the slowest suite. This
runner parallelises at the level of a single (user task, injection task) pair,
which is the granularity at which results are cached and scored.

Repetition isolation. Results are cached at
logdir/pipeline/suite/user_task/attack/injection_task.json and any task already
present is skipped. Three repetitions sharing one logdir would therefore
collapse into one measured pass and two no-ops. Each repetition here gets its
own logdir.

Cost measurement. Nothing in the package records token usage. This runner reads
`usage_metadata` from every API call, prices it against the verified table in
config/models_google.yaml, and enforces a hard ceiling. The budget stops being
an estimate after the first model completes.

Configuration integrity. The run refuses to start when the harness
configuration has moved away from the frozen hash, which is the invariant
deadline pressure is most likely to break.

Resumption is free: re-running the same command skips completed tasks through
AgentDojo's own cache, so an interrupted sweep continues where it stopped.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

try:
    from dotenv import load_dotenv

    load_dotenv(SRC.parent / ".env")
except ImportError:
    pass

import adapters  # noqa: E402
import costs  # noqa: E402
import freeze  # noqa: E402

RESULTS_DIR = SRC.parent / "results"
CONFIG_DIR = SRC.parent / "config"

# The arm's provider module, bound in main(). Every provider exposes the same
# function surface, so selecting an arm binds one module instead of threading a
# branch through the runner. Defaults to Google so an existing command line
# keeps its exact behaviour.
PROV: Any = adapters

ARMS = {
    "google": ("adapters", "models_google.yaml"),
    "baseten": ("adapters_baseten", "models_baseten.yaml"),
    "anthropic": ("adapters_anthropic", "models_anthropic.yaml"),
    "openai": ("adapters_openai", "models_openai.yaml"),
}

_STOP = threading.Event()
_PRINT_LOCK = threading.Lock()


def _pipeline_name(model_id: str) -> str:
    """The name AgentDojo will give the pipeline, and so the results directory.

    Providers whose served ids contain a path separator supply an alias; the
    rest are their own name.
    """
    namer = getattr(PROV, "pipeline_name", None)
    return namer(model_id) if namer else model_id


def _arm_notes(arm: str, config: dict[str, Any]) -> dict[str, Any]:
    """Arm-level adaptations the configuration hash cannot express.

    The hash covers what the model is ASKED: the harness settings, the resolved
    system message text, the attack template, and the task set. It cannot cover
    how a particular provider's wire format differs, and those differences are
    real. Recording them in every manifest keeps a cross-arm comparison honest
    without requiring the reader to diff two adapter modules.
    """
    notes: dict[str, Any] = {
        "max_output_tokens_sent": None,
        "wire_adaptations": [],
    }
    if arm == "google":
        notes["max_output_tokens_sent"] = config["generation"]["max_output_tokens"]
        notes["wire_adaptations"] = [
            "thought_signature relayed on every function-call part, required by Gemini 3.x",
        ]
    elif arm == "baseten":
        notes["max_output_tokens_sent"] = (
            "not sent; AgentDojo's OpenAI path exposes no max-tokens argument, "
            "so each model applies its own default"
        )
        notes["wire_adaptations"] = [
            "system message sent as role=system rather than role=developer; same text, same position",
            "content parts concatenated into a plain string; same characters, different shape",
            "reasoning budget not controllable, so the as_deployed condition is the only one available",
        ]
    elif arm == "anthropic":
        import adapters_anthropic

        notes["max_output_tokens_sent"] = adapters_anthropic.MAX_OUTPUT_TOKENS
        notes["wire_adaptations"] = [
            "max_tokens is REQUIRED by this API and defaults to 1024 in the package; "
            "raised to 8192 so it cannot truncate a turn into a false secure score",
            "as_deployed means extended thinking is OFF, because this API enables it "
            "only on request -- unlike Gemini, where as_deployed means the vendor's own "
            "server-side default. Do not read the two as the same condition",
            "thinking cannot be normalized while holding temperature 0: enabling it "
            "forces temperature=1, so only the as_deployed arm exists here",
            "prompt caching deliberately not enabled, so cached tokens bill at the full "
            "input rate exactly as on the other two arms",
        ]
    return notes


@dataclass
class WorkUnit:
    suite_name: str
    user_task_id: str
    injection_task_id: str | None
    kind: str  # "utility", "injection_utility", or "security"

    @property
    def label(self) -> str:
        if self.injection_task_id is None:
            return f"{self.suite_name}/{self.user_task_id}"
        return f"{self.suite_name}/{self.user_task_id}+{self.injection_task_id}"


class QuietLogger:
    """Delegate satisfying the TraceLogger contract without printing anything.

    A full sweep emits well over a hundred thousand messages. The trace files
    are the record; the console only needs progress.
    """

    def __init__(self, logdir: str) -> None:
        self.logdir = logdir
        self.messages: list[Any] = []
        self.errors: list[str] = []

    def __enter__(self):
        from agentdojo.logging import LOGGER_STACK

        LOGGER_STACK.get().append(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        from agentdojo.logging import LOGGER_STACK

        LOGGER_STACK.get().pop()
        return False

    def log(self, *args, **kwargs) -> None:
        pass

    def log_error(self, message: str) -> None:
        self.errors.append(message)


def _isolate_thread_context() -> None:
    """Give this thread its own logger stack.

    AgentDojo declares `LOGGER_STACK = ContextVar("logger_stack", default=[])`.
    The default is one shared mutable list, so every thread that never calls
    `set` would append to the same object and the stacks would interleave.
    Setting a fresh list per thread isolates them.
    """
    from agentdojo.logging import LOGGER_STACK

    LOGGER_STACK.set([])


def build_work_units(config: dict[str, Any], suites_filter: list[str] | None,
                     max_user_tasks: int | None) -> list[WorkUnit]:
    from agentdojo.task_suite.load_suites import get_suite

    harness = config["harness"]
    benchmark_version = harness["benchmark_version"]
    suite_names = suites_filter or harness["suites"]

    units: list[WorkUnit] = []
    for suite_name in suite_names:
        suite = get_suite(benchmark_version, suite_name)
        user_task_ids = sorted(suite.user_tasks.keys())
        injection_task_ids = sorted(suite.injection_tasks.keys())
        if max_user_tasks is not None:
            user_task_ids = user_task_ids[:max_user_tasks]
            injection_task_ids = injection_task_ids[: max(1, max_user_tasks)]

        # Utility baseline: each user task with no injection.
        for user_task_id in user_task_ids:
            units.append(WorkUnit(suite_name, user_task_id, None, "utility"))

        # Each injection task run as a user task, which AgentDojo uses to check
        # that the injection goal is achievable at all. A security result is
        # only interpretable when the corresponding goal is reachable.
        for injection_task_id in injection_task_ids:
            units.append(WorkUnit(suite_name, injection_task_id, None, "injection_utility"))

        # The security matrix.
        for user_task_id in user_task_ids:
            for injection_task_id in injection_task_ids:
                units.append(WorkUnit(suite_name, user_task_id, injection_task_id, "security"))

    return units


def _run_one(
    unit: WorkUnit,
    suite,
    pipeline,
    attack,
    logdir: Path,
    benchmark_version: str,
    max_retries: int,
) -> dict[str, Any]:
    from agentdojo.benchmark import (
        run_task_with_injection_tasks,
        run_task_without_injection_tasks,
    )

    _isolate_thread_context()
    record = PROV.start_task_usage()

    started = time.time()
    error: str | None = None
    utility = security = None

    for attempt in range(max_retries + 1):
        try:
            with QuietLogger(str(logdir)):
                if unit.injection_task_id is None:
                    if unit.kind == "injection_utility":
                        task = suite.get_injection_task_by_id(unit.user_task_id)
                    else:
                        task = suite.get_user_task_by_id(unit.user_task_id)
                    utility, security = run_task_without_injection_tasks(
                        suite, pipeline, task, logdir, False, benchmark_version
                    )
                else:
                    user_task = suite.get_user_task_by_id(unit.user_task_id)
                    utility_results, security_results = run_task_with_injection_tasks(
                        suite,
                        pipeline,
                        user_task,
                        attack,
                        logdir,
                        False,
                        [unit.injection_task_id],
                        benchmark_version,
                    )
                    key = (unit.user_task_id, unit.injection_task_id)
                    utility = utility_results.get(key)
                    security = security_results.get(key)
            error = None
            break
        except Exception as exc:  # noqa: BLE001 - recorded and retried
            error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries and not _STOP.is_set():
                time.sleep(min(2**attempt * 5, 60))
                continue
            break

    return {
        "suite": unit.suite_name,
        "user_task": unit.user_task_id,
        "injection_task": unit.injection_task_id,
        "kind": unit.kind,
        "utility": utility,
        "security": security,
        "error": error,
        "duration_s": round(time.time() - started, 2),
        "usage": record.as_dict(),
        "_record": record,
    }


def run_model_repetition(
    model_id: str,
    rep: int,
    units: list[WorkUnit],
    config: dict[str, Any],
    client,
    logdir: Path,
    pricing: costs.ModelPricing,
    concurrency: int,
    budget_usd: float,
    spent_so_far: float,
) -> dict[str, Any]:
    from agentdojo.attacks.attack_registry import load_attack
    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
    from agentdojo.task_suite.load_suites import get_suite

    harness = config["harness"]
    generation = config["generation"]
    benchmark_version = harness["benchmark_version"]

    llm = PROV.build_llm(
        model_id,
        client,
        temperature=generation["temperature"],
        max_output_tokens=generation["max_output_tokens"],
    )
    pipeline = AgentPipeline.from_config(
        PipelineConfig(
            llm=llm,
            model_id=None,
            defense=harness["defense"],
            system_message_name=harness["system_message_name"],
            system_message=harness["system_message"],
            tool_output_format=harness["tool_output_format"],
        )
    )
    # from_config derives the name from llm.name; assert it so a future package
    # change cannot silently redirect results into a different tree.
    alias = _pipeline_name(model_id)
    expected_name = alias if harness["defense"] is None else f"{alias}-{harness['defense']}"
    if pipeline.name != expected_name:
        raise RuntimeError(f"Pipeline named {pipeline.name!r}, expected {expected_name!r}")

    suites = {name: get_suite(benchmark_version, name) for name in {u.suite_name for u in units}}
    attacks = {
        name: load_attack(harness["attack"], suite, pipeline) for name, suite in suites.items()
    }

    logdir.mkdir(parents=True, exist_ok=True)
    usage_path = logdir / "usage.jsonl"
    usage_file = usage_path.open("a")

    rep_cost = 0.0
    completed = 0
    errors = 0
    long_context_tasks = 0
    total = len(units)
    started = time.time()
    stopped_for_budget = False

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                _run_one,
                unit,
                suites[unit.suite_name],
                pipeline,
                attacks[unit.suite_name],
                logdir,
                benchmark_version,
                config["execution"]["max_task_retries"],
            ): unit
            for unit in units
        }
        try:
            for future in as_completed(futures):
                result = future.result()
                record = result.pop("_record")
                completed += 1

                task_cost = costs.cost_of_usage(pricing, record)
                rep_cost += task_cost
                result["cost_usd"] = round(task_cost, 6)
                if costs.crossed_long_context(pricing, record):
                    long_context_tasks += 1
                    result["long_context"] = True
                if result["error"]:
                    errors += 1

                usage_file.write(json.dumps(result) + "\n")
                usage_file.flush()

                if completed % 10 == 0 or completed == total:
                    elapsed = time.time() - started
                    rate = completed / elapsed if elapsed else 0
                    remaining = (total - completed) / rate if rate else 0
                    with _PRINT_LOCK:
                        print(
                            f"  {model_id} rep{rep}: {completed}/{total}  "
                            f"{costs.format_usd(rep_cost)}  "
                            f"{errors} errors  "
                            f"{PROV.rate_limit_hits()} rate-limited  "
                            f"eta {remaining / 60:.0f}m",
                            flush=True,
                        )

                if spent_so_far + rep_cost >= budget_usd:
                    stopped_for_budget = True
                    _STOP.set()
                    with _PRINT_LOCK:
                        print(
                            f"\n  BUDGET CEILING of {costs.format_usd(budget_usd)} reached. "
                            f"Cancelling queued tasks.",
                            flush=True,
                        )
                    for pending in futures:
                        pending.cancel()
                    break

                if _STOP.is_set():
                    for pending in futures:
                        pending.cancel()
                    break
        finally:
            usage_file.close()

    return {
        "model": model_id,
        "repetition": rep,
        "logdir": str(logdir),
        "tasks_total": total,
        "tasks_completed": completed,
        "errors": errors,
        "long_context_tasks": long_context_tasks,
        "cost_usd": round(rep_cost, 4),
        "wall_clock_s": round(time.time() - started, 1),
        "stopped_for_budget": stopped_for_budget,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--arm",
        default="google",
        choices=sorted(ARMS),
        help=(
            "Which provider arm to run. Selects the adapter and the model "
            "registry; the frozen harness configuration is shared and unchanged."
        ),
    )
    parser.add_argument("--preset", default=None, help="Model subset from the arm's registry.")
    parser.add_argument("--models", nargs="+", default=None, help="Explicit model ids.")
    parser.add_argument("--suites", nargs="+", default=None, help="Subset of suites.")
    parser.add_argument("--reps", type=int, default=None, help="Overrides execution.repetitions.")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--budget-usd", type=float, default=None)
    parser.add_argument(
        "--max-user-tasks", type=int, default=None, help="Cap user tasks per suite. For smoke runs."
    )
    parser.add_argument("--run-id", default=None, help="Resume by reusing an earlier run id.")
    parser.add_argument(
        "--defense",
        default=None,
        help="Override harness.defense for a defense-toggle condition. Writes to a separate tree.",
    )
    parser.add_argument(
        "--attack",
        default=None,
        help=(
            "Override harness.attack. Used for the contamination probe, where a "
            "variant preserves the attack's semantics while changing its surface "
            "form. The attack is part of the frozen configuration, so this also "
            "requires --allow-config-change and marks the results as a separate "
            "configuration generation. Always include a known-vulnerable model as "
            "a positive control: a variant that scores zero everywhere is a weak "
            "attack, not evidence of hardening."
        ),
    )
    parser.add_argument(
        "--thinking-mode",
        default=None,
        choices=["as_deployed", "normalized", "off"],
        help="Override generation.thinking_mode.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "Override generation.temperature. Needed because a vendor may remove "
            "the control: claude-sonnet-5 rejects the temperature field outright "
            "('deprecated for this model') while its two predecessors accept 0.0, "
            "so the lineage can only be compared at that model's default of 1.0. "
            "Temperature is inside the frozen fingerprint, so this requires "
            "--allow-config-change and marks the results as a separate "
            "configuration generation. Setting it above 0 costs determinism, which "
            "means the repetition floor stops being optional."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Plan and project cost without calling the API."
    )
    parser.add_argument(
        "--allow-config-change",
        action="store_true",
        help="Proceed despite a changed configuration hash. Results are marked as a different generation.",
    )
    args = parser.parse_args()

    global PROV
    module_name, registry_filename = ARMS[args.arm]
    PROV = __import__(module_name)
    registry_path = CONFIG_DIR / registry_filename

    config = freeze.load_harness_config()
    if args.defense is not None:
        config["harness"]["defense"] = None if args.defense == "none" else args.defense
    if args.attack is not None:
        config["harness"]["attack"] = args.attack
    if args.thinking_mode is not None:
        config["generation"]["thinking_mode"] = args.thinking_mode
    if args.temperature is not None:
        config["generation"]["temperature"] = args.temperature
    if args.concurrency is not None:
        config["execution"]["concurrency"] = args.concurrency
    if args.budget_usd is not None:
        config["execution"]["budget_usd"] = args.budget_usd
    if args.reps is not None:
        config["execution"]["repetitions"] = args.reps

    # ---- configuration integrity ------------------------------------------
    # Hash the effective configuration, so a command-line override is caught
    # rather than certified away.
    matches, current_hash, locked_hash = freeze.check(config=config)
    if locked_hash is None:
        print("Harness configuration is not frozen. Run `python src/freeze.py write` first.")
        print(f"Current hash would be {current_hash}.")
        return 1
    if not matches and not args.allow_config_change:
        print("REFUSING TO RUN: the harness configuration has changed since it was frozen.")
        print(f"  locked  : {locked_hash}")
        print(f"  current : {current_hash}")
        print()
        print("Cross-arm comparison depends on every arm sharing one configuration.")
        print("Run `python src/freeze.py check` to see what moved, then either revert")
        print("the change or re-freeze and re-run every arm.")
        print("Pass --allow-config-change only if you intend a new configuration generation.")
        return 1

    registry = costs.load_registry(registry_path)
    pricing_table = costs.load_pricing(registry_path)

    if args.models:
        model_ids = args.models
    elif args.preset:
        if args.preset not in registry["presets"]:
            print(f"Unknown preset {args.preset!r}. Known: {sorted(registry['presets'])}")
            return 1
        model_ids = registry["presets"][args.preset]["models"]
    else:
        print("Specify --preset or --models. Presets available:")
        for name, spec in registry["presets"].items():
            print(f"  {name:<20} {spec['description']}")
        return 1

    unknown = [m for m in model_ids if m not in pricing_table]
    if unknown:
        print(f"No pricing for: {', '.join(unknown)}. Add them to config/{registry_filename}.")
        return 1

    units = build_work_units(config, args.suites, args.max_user_tasks)
    reps = config["execution"]["repetitions"]
    concurrency = config["execution"]["concurrency"]
    budget = config["execution"]["budget_usd"]

    print("=" * 78)
    print(f"AGENTDOJO SWEEP: {args.arm.upper()} ARM")
    print("=" * 78)
    print(f"config_hash      {current_hash}" + ("" if matches else "  (CHANGED, forced)"))
    print(f"benchmark        {config['harness']['benchmark_version']}")
    print(f"attack           {config['harness']['attack']}")
    print(f"defense          {config['harness']['defense']}")
    print(f"thinking         {config['generation']['thinking_mode']}")
    print(f"models           {len(model_ids)}: {', '.join(model_ids)}")
    print(f"tasks per rep    {len(units)}")
    print(f"repetitions      {reps}")
    print(f"total task runs  {len(units) * reps * len(model_ids):,}")
    print(f"concurrency      {concurrency}")
    print(f"budget ceiling   {costs.format_usd(budget)}")

    # Projection from the published AgentDojo anchor, stated as an estimate
    # until the first repetition replaces it with a measurement.
    print("\nProjected cost, from the published AgentDojo anchor of about")
    print("$39 per pass on GPT-4o at $2.50 per million input tokens, scaled by")
    print("input price and by the ratio of this task set to the 753 runs that")
    print("anchor covers. Treat as an estimate until the first repetition lands.")
    anchor_cost, anchor_runs, anchor_input_price = 39.0, 753, 2.50
    scale_tasks = len(units) / anchor_runs
    projected_total = 0.0
    print(f"\n  {'model':<28}{'in $/M':>8}{'per rep':>11}{'x' + str(reps):>11}")
    print("  " + "-" * 58)
    for model_id in model_ids:
        spec = pricing_table[model_id]
        per_rep = anchor_cost * scale_tasks * (spec.input / anchor_input_price)
        projected_total += per_rep * reps
        print(
            f"  {model_id:<28}{spec.input:>8.2f}"
            f"{costs.format_usd(per_rep):>11}{costs.format_usd(per_rep * reps):>11}"
        )
    print("  " + "-" * 58)
    print(f"  {'projected total':<28}{'':>8}{'':>11}{costs.format_usd(projected_total):>11}")
    if projected_total > budget:
        print(f"\n  WARNING: projection exceeds the ceiling of {costs.format_usd(budget)}.")
        print("  The sweep will stop cleanly at the ceiling and remain resumable.")

    if args.dry_run:
        print("\nDry run, nothing called.")
        return 0

    # ---- go ----------------------------------------------------------------
    try:
        PROV.install(
            thinking_mode=config["generation"]["thinking_mode"],
            thinking_budget=config["generation"]["thinking_budget"],
        )
    except ValueError as exc:
        print(f"\n{exc}")
        return 1
    PROV.register_model_names(model_ids)
    try:
        client = PROV.build_client()
    except RuntimeError as exc:
        print(f"\n{exc}")
        return 1

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RESULTS_DIR / f"{args.arm}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nresults -> {run_dir}")
    if args.run_id:
        print("Resuming: completed tasks are skipped through AgentDojo's cache.")

    def _handle_sigint(_signum, _frame):
        with _PRINT_LOCK:
            print("\nInterrupt received. Finishing in-flight tasks, then stopping.")
            print("Re-run with --run-id " + run_id + " to resume.")
        _STOP.set()

    signal.signal(signal.SIGINT, _handle_sigint)

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "arm": args.arm,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": current_hash,
        "config_matches_lock": matches,
        "harness": config["harness"],
        "generation": config["generation"],
        "execution": {
            "repetitions": reps,
            "concurrency": concurrency,
            "budget_usd": budget,
        },
        "models": model_ids,
        # Served id -> the directory and pipeline name its traces are written
        # under. Identical on arms whose ids need no aliasing; recorded always,
        # so nothing has to infer the mapping from the tree later.
        "model_aliases": {m: _pipeline_name(m) for m in model_ids},
        "tasks_per_repetition": len(units),
        "suites": args.suites or config["harness"]["suites"],
        "max_user_tasks": args.max_user_tasks,
        "pricing_verified_on": registry["verified_on"],
        "model_registry": registry_filename,
        "base_url": registry.get("base_url"),
        # Arm-level adaptations that the configuration hash does not cover,
        # recorded so a cross-arm reader sees them without reading the adapters.
        "arm_notes": _arm_notes(args.arm, config),
        "repetition_results": [],
    }

    def _save_manifest() -> None:
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    _save_manifest()

    total_spent = 0.0
    wall_start = time.time()
    for model_id in model_ids:
        if _STOP.is_set():
            break
        for rep in range(1, reps + 1):
            if _STOP.is_set():
                break
            logdir = run_dir / _pipeline_name(model_id) / f"rep{rep}"
            print(f"\n--- {model_id}, repetition {rep} of {reps} ---")
            summary = run_model_repetition(
                model_id=model_id,
                rep=rep,
                units=units,
                config=config,
                client=client,
                logdir=logdir,
                pricing=pricing_table[model_id],
                concurrency=concurrency,
                budget_usd=budget,
                spent_so_far=total_spent,
            )
            total_spent += summary["cost_usd"]
            summary["cumulative_cost_usd"] = round(total_spent, 4)
            manifest["repetition_results"].append(summary)
            _save_manifest()
            print(
                f"  done: {summary['tasks_completed']}/{summary['tasks_total']} tasks, "
                f"{costs.format_usd(summary['cost_usd'])}, "
                f"{summary['wall_clock_s'] / 60:.0f}m, "
                f"{summary['errors']} errors"
            )
            if summary["long_context_tasks"]:
                print(
                    f"  note: {summary['long_context_tasks']} tasks crossed the "
                    f"200k long-context pricing tier"
                )
            if summary["stopped_for_budget"]:
                break

    totals = PROV.global_usage()
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["total_cost_usd"] = round(total_spent, 4)
    manifest["wall_clock_s"] = round(time.time() - wall_start, 1)
    manifest["token_totals"] = totals.as_dict()
    manifest["rate_limit_hits"] = PROV.rate_limit_hits()
    manifest["stopped_early"] = _STOP.is_set()
    _save_manifest()

    print("\n" + "=" * 78)
    print(f"measured cost      {costs.format_usd(total_spent)}")
    print(f"prompt tokens      {totals.prompt_tokens:,}")
    print(f"output tokens      {totals.output_tokens:,}  (+{totals.thought_tokens:,} thinking)")
    print(f"cached tokens      {totals.cached_tokens:,}  (priced at full rate, so cost is an upper bound)")
    print(f"api calls          {totals.calls:,}")
    print(f"rate-limit hits    {PROV.rate_limit_hits()}")
    print(f"wall clock         {manifest['wall_clock_s'] / 3600:.1f}h")
    print(f"manifest           {run_dir / 'manifest.json'}")
    if _STOP.is_set():
        print(f"\nStopped early. Resume with:  --run-id {run_id}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
