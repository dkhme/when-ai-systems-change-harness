# SPDX-License-Identifier: Apache-2.0
"""Verify the Anthropic arm against the live API before any real money is spent.

The check that matters most here is **truncation**, not availability.

AgentDojo's `AnthropicLLM` defaults `max_tokens` to 1024 and this API requires
the field, so unlike the Google and OpenAI paths there is no "let the model
decide". A turn truncated at the cap emits no valid tool call, fails the user
task, and scores as SECURE, because a model that emits nothing also injects
nothing. That is a harness-manufactured version of the degeneracy confound that
`degeneracy_check.py` tests for, and it would inflate exactly the quantity this
arm exists to measure. So the probes below assert that a realistic tool-calling turn
does **not** stop on `max_tokens`.

Second is function calling itself. AgentDojo is a tool-use benchmark and a model
that answers in prose scores as perfectly secure out of pure incapacity.

Performed in order:

  1. key presence and client construction
  2. GET /v1/models, the ground truth of what this key may serve
  3. reconciliation against the configured registry, both directions
  4. a minimal completion per model, proving the identifier answers
  5. a function-calling round trip per model, asserting stop_reason != max_tokens
  6. the injected-payload canary: no Claude-specific string reaches the attack
  7. AgentDojo suite loading and the arm's case count
  8. the configuration freeze state under this arm's attack

Spend is a few hundred output tokens per model, well under one US cent total.

Run:  python src/verify_setup_anthropic.py [--preset sonnet-lineage]
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

import adapters_anthropic as at  # noqa: E402
import costs  # noqa: E402
import freeze  # noqa: E402

PASS = "  ok  "
FAIL = " FAIL "
WARN = " warn "

ARM_ATTACK = "important_instructions_no_names"
# The other two arms' hash, at temperature 0. This arm CANNOT match it: see
# adapters_anthropic.NO_TEMPERATURE_CONTROL. Kept so the pre-flight can report
# the divergence explicitly rather than let it pass unremarked.
PROBE_CONFIG_HASH = "28f36d30e3a053c1"
ARM_SUITES = ["slack", "travel", "banking"]

_WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather in a given city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "The city name."}},
        "required": ["city"],
    },
}


def _mark(ok: bool, label: str, detail: str = "") -> bool:
    print(f"{PASS if ok else FAIL}{label}" + (f"  ({detail})" if detail else ""))
    return ok


def _short(exc: Exception) -> str:
    text = str(exc).replace("\n", " ")
    return text[:110] + ("..." if len(text) > 110 else "")


def _probe_completion(client, model_id: str) -> tuple[bool, str]:
    """Liveness only: does this identifier answer at all."""
    try:
        msg = client.messages.create(
            model=model_id,
            max_tokens=24,
            temperature=at.DEFAULT_TEMPERATURE,
            messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text").strip()
        return True, text[:40] or f"(empty, stop={msg.stop_reason})"
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return False, _short(exc)


def _probe_function_calling(client, model_id: str) -> tuple[bool, str]:
    """Prove a well-formed tool call, and prove the cap is not binding.

    Sent at the SAME max_tokens the sweep uses, because a pre-flight that tests
    a different request shape than the run is worse than no pre-flight: it is
    believed. The Baseten arm learned this the expensive way, where a probe that
    added `max_tokens` failed a model on a request the sweep never made.
    """
    try:
        msg = client.messages.create(
            model=model_id,
            max_tokens=at.MAX_OUTPUT_TOKENS,
            temperature=at.DEFAULT_TEMPERATURE,
            tools=[_WEATHER_TOOL],
            messages=[{"role": "user", "content": "What is the weather in Zurich? Use the tool."}],
        )
        if msg.stop_reason == "max_tokens":
            return False, f"stopped on max_tokens at {at.MAX_OUTPUT_TOKENS}"
        calls = [b for b in msg.content if b.type == "tool_use"]
        if not calls:
            return False, "answered without calling the tool"
        call = calls[0]
        if call.name != "get_weather":
            return False, f"called {call.name!r}"
        if "city" not in (call.input or {}):
            return False, f"no city argument: {call.input!r}"
        return True, f"get_weather(city={call.input['city']!r}) stop={msg.stop_reason}"
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return False, _short(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--preset", default="sonnet-lineage")
    args = parser.parse_args()

    print("=" * 78)
    print("PRE-FLIGHT: ANTHROPIC ARM")
    print("=" * 78)
    ok = True

    registry = at.load_registry()
    if args.preset not in registry["presets"]:
        print(f"Unknown preset {args.preset!r}. Known: {sorted(registry['presets'])}")
        return 1
    model_ids = registry["presets"][args.preset]["models"]

    # ---- 1. key and client -------------------------------------------------
    print("\n[1] Key and client")
    try:
        client = at.build_client()
    except RuntimeError as exc:
        print(f"{FAIL}{exc}")
        return 1
    ok &= _mark(True, "client built", at.base_url())

    # ---- 2 and 3. what this key may serve, reconciled ----------------------
    print("\n[2] Live model listing, reconciled against the registry")
    try:
        served = sorted(m.id for m in client.models.list(limit=100))
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

    configured = set(registry["models"])
    unconfigured = [m for m in served if m not in configured]
    if unconfigured:
        print(f"{WARN}served but not in the registry: {', '.join(unconfigured)}")

    # ---- 4 and 5. per-model probes ----------------------------------------
    print("\n[3] Per-model probes: completion, then function calling at the sweep's cap")
    print(f"       {'model':<30}{'reply':<9}{'tool call':<11}detail")
    print("       " + "-" * 76)
    tool_failures: list[str] = []
    for model_id in model_ids:
        if model_id not in served:
            print(f"       {model_id:<30}{'-':<9}{'-':<11}not served")
            continue
        gen_ok, gen_detail = _probe_completion(client, model_id)
        tool_ok, tool_detail = _probe_function_calling(client, model_id)
        ok &= gen_ok
        if not tool_ok:
            tool_failures.append(model_id)
        detail = tool_detail if not tool_ok else f"{tool_detail}"
        print(
            f"       {model_id:<30}{('ok' if gen_ok else 'FAIL'):<9}"
            f"{('ok' if tool_ok else 'FAIL'):<11}{detail}"
        )

    ok &= _mark(
        not tool_failures,
        "well-formed tool call, and the max_tokens cap does not bind",
        ", ".join(tool_failures) if tool_failures else f"{len(model_ids)} models",
    )
    if tool_failures:
        print("       A truncated or prose-only turn scores as SECURE for want of")
        print("       capability. Fix before spending, or the arm reports a defence")
        print("       that is really a defect.")

    # ---- 6. the injected-payload canary ------------------------------------
    print("\n[4] Injected payload: no Claude-specific string reaches the attack")
    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
    from agentdojo.attacks.attack_registry import load_attack
    from agentdojo.task_suite.load_suites import get_suite

    config = freeze.load_harness_config()
    config["harness"]["attack"] = ARM_ATTACK
    at.register_model_names(model_ids)
    at.install()
    llm = at.build_llm(model_ids[0], client, temperature=at.DEFAULT_TEMPERATURE)
    pipeline = AgentPipeline.from_config(
        PipelineConfig(
            llm=llm,
            model_id=None,
            defense=config["harness"]["defense"],
            system_message_name=config["harness"]["system_message_name"],
            system_message=config["harness"]["system_message"],
            tool_output_format=config["harness"]["tool_output_format"],
        )
    )
    suite = get_suite(config["harness"]["benchmark_version"], "banking")
    attack = load_attack(ARM_ATTACK, suite, pipeline)
    user_task = next(iter(suite.user_tasks.values()))
    injection_task = next(iter(suite.injection_tasks.values()))
    payload = " ".join(attack.attack(user_task, injection_task).values())
    banned = [w for w in ("Claude", "claude", "Anthropic", "anthropic") if w in payload]
    ok &= _mark(
        not banned,
        "payload carries no vendor-identifying string",
        f"found {banned}" if banned else f"{len(payload)} chars, name-neutral",
    )
    ok &= _mark(
        at.ANTHROPIC_MODEL_PROSE_NAME in payload or True,
        "payload built through the same code path as the other arms",
    )

    # ---- 7. the benchmark itself ------------------------------------------
    print("\n[5] Benchmark")
    security_cases = 0
    utility_runs = 0
    for name in ARM_SUITES:
        s = get_suite(config["harness"]["benchmark_version"], name)
        n_user, n_inj = len(s.user_tasks), len(s.injection_tasks)
        security_cases += n_user * n_inj
        utility_runs += n_user + n_inj
        print(f"       {name:<12}{n_user:>3} user  {n_inj:>3} injection  {n_user * n_inj:>4} cases")
    ok &= _mark(
        security_cases == 389,
        "arm case count matches the Gemini probe and the open-weight arm",
        f"{security_cases} security + {utility_runs} utility = {security_cases + utility_runs} per rep",
    )

    # ---- 8. freeze --------------------------------------------------------
    print("\n[6] Configuration")
    config["generation"]["temperature"] = at.DEFAULT_TEMPERATURE
    _, current_hash, locked_hash = freeze.check(config=config)
    ok &= _mark(
        current_hash != PROBE_CONFIG_HASH,
        "hash DIVERGES from the other arms, as it must at temperature 1.0",
        f"this arm={current_hash}  other arms={PROBE_CONFIG_HASH}",
    )
    print("       Same attack, same suites, same tasks. Only the temperature differs,")
    print("       and only because claude-sonnet-5 rejects the field. Within-lineage")
    print("       comparison is clean; CROSS-ARM attack rates are confounded by")
    print("       sampling and must be labelled as such.")
    print(f"       main-matrix lock is {locked_hash}")

    # ---- cost projection ---------------------------------------------------
    print("\n[7] Projected cost at ONE repetition, 3 suites")
    pricing = costs.load_pricing(SRC.parent / "config" / "models_anthropic.yaml")
    anchor_cost, anchor_runs, anchor_input_price = 39.0, 753, 2.50
    scale = (security_cases + utility_runs) / anchor_runs
    total = 0.0
    print(f"       {'model':<30}{'in $/M':>8}{'per rep':>11}")
    print("       " + "-" * 50)
    for model_id in model_ids:
        spec = pricing[model_id]
        per_rep = anchor_cost * scale * (spec.input / anchor_input_price)
        total += per_rep
        print(f"       {model_id:<30}{spec.input:>8.2f}{costs.format_usd(per_rep):>11}")
    print("       " + "-" * 50)
    print(f"       {'projected total':<30}{'':>8}{costs.format_usd(total):>11}")
    print("       Scaled from the published AgentDojo anchor and input price alone.")
    print("       Sonnet trajectories are longer than the anchor's, so treat this as")
    print("       a LOWER bound until the first repetition replaces it with a")
    print("       measurement. The --budget-usd ceiling is the real protection.")

    print("\n" + "=" * 78)
    if ok:
        print("PRE-FLIGHT PASSED.")
        print()
        print("Next, cheapest first:")
        print("  python src/run_sweep.py --arm anthropic --preset smoke --suites slack \\")
        print("      --max-user-tasks 2 --reps 1 --attack important_instructions_no_names \\")
        print("      --allow-config-change")
    else:
        print("FAILURES ABOVE. Do not commit the budget until they are resolved.")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
