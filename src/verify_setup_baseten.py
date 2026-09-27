# SPDX-License-Identifier: Apache-2.0
"""Verify the Baseten arm against the live API before any real money is spent.

`config/models_baseten.yaml` was checked against Baseten's pricing page and a
live model listing on 9 August 2026, but a page is not an endpoint and a served
id can be withdrawn between an edit and a run. Three of the eight ids this
provider served on 2 July 2026 were already gone five weeks later, so the prior
on a stale registry here is high rather than low.

The check that actually matters on this arm is function calling. AgentDojo is a
tool-use benchmark and is useless without it; open-weight models vary in whether
they emit well-formed tool calls through an OpenAI-compatible endpoint at all,
and a model that cannot call a tool scores as perfectly secure for want of
capability. Discovering that after committing the budget would waste the run.

Performed in order:

  1. key presence and client construction
  2. GET /v1/models, the ground truth of what this key may serve
  3. reconciliation against the configured registry, both directions
  4. a minimal completion per model, proving the identifier answers
  5. a minimal function-calling round trip per model
  6. AgentDojo suite loading and the arm's case count
  7. the configuration freeze state under this arm's attack

Spend is capped at a few dozen output tokens per model, well under one US cent
for the whole set.

Run:  python src/verify_setup_baseten.py [--preset mvc]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

try:
    from dotenv import load_dotenv

    load_dotenv(SRC.parent / ".env")
except ImportError:
    pass

import adapters_baseten as bt  # noqa: E402
import costs  # noqa: E402
import freeze  # noqa: E402

PASS = "  ok  "
FAIL = " FAIL "
WARN = " warn "

ARM_ATTACK = "important_instructions_no_names"
PROBE_CONFIG_HASH = "28f36d30e3a053c1"
ARM_SUITES = ["slack", "travel", "banking"]

# Kept tiny: these probes exist to prove a path, not to measure anything.
PROBE_MAX_TOKENS = 24

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "The city name."}},
            "required": ["city"],
        },
    },
}


def _mark(ok: bool, label: str, detail: str = "") -> bool:
    print(f"{PASS if ok else FAIL}{label}" + (f"  ({detail})" if detail else ""))
    return ok


def _short(exc: Exception) -> str:
    text = str(exc).replace("\n", " ")
    return text[:110] + ("..." if len(text) > 110 else "")


def _probe_completion(client, model_id: str) -> tuple[bool, str]:
    """Liveness only: does this identifier answer at all.

    Capped with `max_tokens` because it is a liveness check and cost is the only
    thing at stake. A reasoning model may spend the whole cap on reasoning and
    return empty text with finish_reason "length"; that still proves the
    identifier answers, which is all this probe claims.
    """
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Reply with the single word: ready"}],
            temperature=0.0,
            max_tokens=PROBE_MAX_TOKENS,
        )
        text = (response.choices[0].message.content or "").strip()
        if text:
            return True, text[:40]
        return True, f"(empty, finish={response.choices[0].finish_reason})"
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return False, _short(exc)


def _probe_function_calling(client, model_id: str) -> tuple[bool, str]:
    """Prove the model emits a well-formed tool call through this endpoint.

    Accepts only a parseable call to the offered function with the required
    argument present. A model that answers in prose here cannot run AgentDojo,
    and would otherwise post a perfect security score built entirely on
    incapacity.

    THE REQUEST SHAPE HERE MUST MIRROR THE SWEEP'S. AgentDojo's OpenAI path
    sends model, messages, tools, tool_choice, temperature and reasoning_effort,
    and NO max_tokens. An earlier version of this probe added `max_tokens` to
    keep the check cheap, and `moonshotai/Kimi-K3` answered 500 to every
    tool-calling request that carried it while succeeding without it. The probe
    had failed a model on a request the sweep never makes. A pre-flight that
    tests a different shape than the run is worse than no pre-flight, because it
    is believed. Cost is bounded by the reply being a single tool call.
    """
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "What is the weather in Zurich? Use the tool."}],
            tools=[_WEATHER_TOOL],
            tool_choice="auto",
            temperature=0.0,
        )
        message = response.choices[0].message
        calls = getattr(message, "tool_calls", None)
        if not calls:
            return False, "answered without calling the tool"
        call = calls[0]
        if call.function.name != "get_weather":
            return False, f"called {call.function.name!r}"
        args = json.loads(call.function.arguments or "{}")
        if "city" not in args:
            return False, f"no city argument: {call.function.arguments!r}"
        return True, f"get_weather(city={args['city']!r})"
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return False, _short(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--preset",
        default="mvc",
        help="Which subset of config/models_baseten.yaml to verify. Default: mvc.",
    )
    args = parser.parse_args()

    print("=" * 78)
    print("PRE-FLIGHT: BASETEN ARM")
    print("=" * 78)
    ok = True

    registry = bt.load_registry()
    if args.preset not in registry["presets"]:
        print(f"Unknown preset {args.preset!r}. Known: {sorted(registry['presets'])}")
        return 1
    model_ids = registry["presets"][args.preset]["models"]

    # ---- 1. key and client -------------------------------------------------
    print("\n[1] Key and client")
    try:
        client = bt.build_client()
    except RuntimeError as exc:
        print(f"{FAIL}{exc}")
        return 1
    ok &= _mark(True, "client built", bt.base_url())

    # ---- 2 and 3. what this key may serve, reconciled ----------------------
    print("\n[2] Live model listing, reconciled against the registry")
    try:
        served = sorted(m.id for m in client.models.list().data)
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL}models.list failed: {_short(exc)}")
        return 1
    print(f"       {len(served)} ids served by this key")

    missing = [m for m in model_ids if m not in served]
    ok &= _mark(
        not missing,
        "every model in the preset is served",
        ", ".join(missing) if missing else f"{len(model_ids)} models",
    )
    if missing:
        print("       A configured id is no longer served. Update")
        print("       config/models_baseten.yaml before spending, and record the")
        print("       withdrawal date; the catalogue moves on this provider.")

    configured = set(registry["models"])
    unconfigured = [m for m in served if m not in configured]
    if unconfigured:
        print(f"{WARN}served but not in the registry: {', '.join(unconfigured)}")

    gone = [m for m in registry.get("not_served_here_since_2026_07_02", []) if m in served]
    if gone:
        print(f"{WARN}listed as withdrawn but served again: {', '.join(gone)}")

    # ---- 4 and 5. per-model probes ----------------------------------------
    print("\n[3] Per-model probes: completion, then function calling")
    print(f"       {'model':<30}{'reply':<10}{'tool call':<12}detail")
    print("       " + "-" * 68)
    tool_failures: list[str] = []
    for model_id in model_ids:
        if model_id not in served:
            print(f"       {model_id:<30}{'-':<10}{'-':<12}not served")
            continue
        gen_ok, gen_detail = _probe_completion(client, model_id)
        tool_ok, tool_detail = _probe_function_calling(client, model_id)
        ok &= gen_ok
        if not tool_ok:
            tool_failures.append(model_id)
        detail = tool_detail if not tool_ok else gen_detail
        print(
            f"       {model_id:<30}{('ok' if gen_ok else 'FAIL'):<10}"
            f"{('ok' if tool_ok else 'FAIL'):<12}{detail}"
        )

    ok &= _mark(
        not tool_failures,
        "every model emits a well-formed tool call",
        ", ".join(tool_failures) if tool_failures else f"{len(model_ids)} models",
    )
    if tool_failures:
        print("       A model that cannot call a tool cannot run AgentDojo, and")
        print("       would post a perfect security score out of pure incapacity.")
        print("       Drop it from the preset, or the arm reports a defence that")
        print("       is really a defect.")

    # ---- 6. the benchmark itself ------------------------------------------
    print("\n[4] Benchmark")
    config = freeze.load_harness_config()
    config["harness"]["attack"] = ARM_ATTACK
    from agentdojo.task_suite.load_suites import get_suite

    security_cases = 0
    utility_runs = 0
    for name in ARM_SUITES:
        suite = get_suite(config["harness"]["benchmark_version"], name)
        n_user, n_inj = len(suite.user_tasks), len(suite.injection_tasks)
        security_cases += n_user * n_inj
        utility_runs += n_user + n_inj
        print(f"       {name:<12}{n_user:>3} user  {n_inj:>3} injection  {n_user * n_inj:>4} cases")
    ok &= _mark(
        security_cases == 389,
        "arm case count matches the Gemini contamination probe",
        f"{security_cases} security + {utility_runs} utility = {security_cases + utility_runs} runs per rep",
    )

    # ---- 7. freeze --------------------------------------------------------
    print("\n[5] Configuration")
    _, current_hash, locked_hash = freeze.check(config=config)
    ok &= _mark(
        current_hash == PROBE_CONFIG_HASH,
        "hash matches the Gemini probe, so the arms are comparable",
        f"current={current_hash} probe={PROBE_CONFIG_HASH}",
    )
    print(f"       main-matrix lock is {locked_hash}; this arm is a separate generation")

    # ---- cost projection ---------------------------------------------------
    print("\n[6] Projected cost at 2 repetitions, 3 suites")
    pricing = costs.load_pricing(SRC.parent / "config" / "models_baseten.yaml")
    anchor_cost, anchor_runs, anchor_input_price = 39.0, 753, 2.50
    scale = (security_cases + utility_runs) / anchor_runs
    total = 0.0
    print(f"       {'model':<30}{'in $/M':>8}{'per rep':>11}{'x2':>11}")
    print("       " + "-" * 60)
    for model_id in model_ids:
        spec = pricing[model_id]
        per_rep = anchor_cost * scale * (spec.input / anchor_input_price)
        total += per_rep * 2
        print(
            f"       {model_id:<30}{spec.input:>8.2f}"
            f"{costs.format_usd(per_rep):>11}{costs.format_usd(per_rep * 2):>11}"
        )
    print("       " + "-" * 60)
    print(f"       {'projected total':<30}{'':>8}{'':>11}{costs.format_usd(total):>11}")
    print("       Scaled from the published AgentDojo anchor and input price alone.")
    print("       Open-weight trajectories may be longer, so treat it as a lower")
    print("       bound until the first repetition replaces it with a measurement.")

    print("\n" + "=" * 78)
    if ok:
        print("PRE-FLIGHT PASSED.")
        print()
        print("Next, cheapest first:")
        print("  python src/run_sweep.py --arm baseten --preset smoke --suites slack \\")
        print("      --max-user-tasks 2 --reps 1 --attack important_instructions_no_names \\")
        print("      --allow-config-change")
    else:
        print("FAILURES ABOVE. Do not commit the budget until they are resolved.")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
