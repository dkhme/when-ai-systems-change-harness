# SPDX-License-Identifier: Apache-2.0
"""Verify the environment before any money is spent.

Model identifiers and prices in config/models_google.yaml were verified against
Google's published documentation on 7 August 2026, but a document is not the API
and an identifier can be withdrawn between a page edit and a run. A wrong
identifier is the immediate blocker on the Google arm, for it wastes the window
rather than the money.

This script resolves the question against the live API using the key that will
run the sweep. It performs, in order:

  1. key presence and client construction
  2. models.list, the ground truth of what this key may serve
  3. reconciliation against the configured registry
  4. a minimal generate call per model, proving the identifier answers
  5. a minimal function-calling call per model, proving tool use works, since
     AgentDojo is useless without it
  6. AgentDojo import, suite loading, and task counts
  7. the configuration freeze state

Total spend is a few hundred tokens per model, under one US cent for the whole
matrix. Run it whenever the key, the model set, or the machine changes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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

OK = "  ok "
FAIL = "FAIL "
WARN = "warn "


def _probe_generate(client, model_id: str) -> tuple[bool, str]:
    """Smallest possible generation, to prove the identifier is servable."""
    from google.genai import types as genai_types

    try:
        response = client.models.generate_content(
            model=model_id,
            contents="Reply with the single word: ready",
            config=genai_types.GenerateContentConfig(max_output_tokens=2048),
        )
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return False, _short_error(exc)

    usage = getattr(response, "usage_metadata", None)
    tokens = getattr(usage, "total_token_count", 0) or 0
    return True, f"{tokens} tokens"


def _probe_function_calling(client, model_id: str) -> tuple[bool, str]:
    """Prove the model emits a function call, which AgentDojo depends on."""
    from google.genai import types as genai_types

    tool = genai_types.Tool(
        function_declarations=[
            genai_types.FunctionDeclaration(
                name="get_balance",
                description="Return the account balance for an account id.",
                parameters=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={
                        "account_id": genai_types.Schema(
                            type=genai_types.Type.STRING,
                            description="The account identifier.",
                        )
                    },
                    required=["account_id"],
                ),
            )
        ]
    )
    try:
        response = client.models.generate_content(
            model=model_id,
            contents="What is the balance of account ABC-123? Use the tool.",
            config=genai_types.GenerateContentConfig(
                tools=[tool],
                max_output_tokens=4096,
                temperature=0.0,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return False, _short_error(exc)

    if not response.candidates or response.candidates[0].content is None:
        return False, "no candidate returned"
    parts = response.candidates[0].content.parts or []
    for part in parts:
        if getattr(part, "function_call", None) is not None:
            return True, f"called {part.function_call.name}"
    return False, "model answered without calling the tool"


def _short_error(exc: Exception) -> str:
    text = str(exc).replace("\n", " ")
    if len(text) > 90:
        text = text[:87] + "..."
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--preset",
        default="all",
        help="Which model subset from models_google.yaml to verify. Default: all.",
    )
    parser.add_argument(
        "--skip-probes",
        action="store_true",
        help="Skip the live generate and tool-call probes. Costs nothing, proves less.",
    )
    args = parser.parse_args()

    failures = 0
    warnings = 0

    print("=" * 78)
    print("GEMINI ARM: PRE-FLIGHT VERIFICATION")
    print("=" * 78)

    # ---- 1. key and client -------------------------------------------------
    print("\n[1] API key and client")
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        print(f"{FAIL} No GEMINI_API_KEY or GOOGLE_API_KEY found.")
        print("      Copy .env.example to .env at the repository root and paste the key.")
        return 1
    print(f"{OK} key present, {len(key)} characters, ending {key[-4:]}")

    try:
        client = adapters.build_client()
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} client construction failed: {_short_error(exc)}")
        return 1
    print(f"{OK} google-genai client constructed against the Developer API")

    # ---- 2. what this key can actually serve -------------------------------
    print("\n[2] Models visible to this key")
    try:
        served = set()
        for model in client.models.list():
            name = (model.name or "").removeprefix("models/")
            if name:
                served.add(name)
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} models.list failed: {_short_error(exc)}")
        return 1
    print(f"{OK} {len(served)} models visible")

    # ---- 3. reconcile against the registry --------------------------------
    print("\n[3] Configured models versus the live list")
    registry = costs.load_registry()
    presets = registry["presets"]
    if args.preset not in presets:
        print(f"{FAIL} unknown preset {args.preset!r}. Known: {sorted(presets)}")
        return 1
    configured = presets[args.preset]["models"]

    pricing = costs.load_pricing()
    print(f"\n     {'model':<28}{'lineage':<12}{'listed':<9}{'in $/M':>8}{'out $/M':>9}")
    print("     " + "-" * 66)
    missing = []
    for model_id in configured:
        spec = pricing[model_id]
        listed = model_id in served
        if not listed:
            missing.append(model_id)
        marker = "yes" if listed else "NO"
        print(
            f"     {model_id:<28}{spec.lineage:<12}{marker:<9}"
            f"{spec.input:>8.2f}{spec.output:>9.2f}"
        )

    if missing:
        failures += 1
        print(f"\n{FAIL} not visible to this key: {', '.join(missing)}")
        print("      An identifier may have been withdrawn, or the key may lack access.")
        print("      Check https://ai.google.dev/gemini-api/docs/models and update")
        print("      config/models_google.yaml before running the sweep.")
    else:
        print(f"\n{OK} every configured model in preset '{args.preset}' is servable")

    # Report anything new that the registry has not yet caught up with.
    interesting = sorted(
        name
        for name in served
        if name.startswith("gemini-")
        and name not in pricing
        and not any(
            token in name
            for token in ("embedding", "tts", "audio", "image", "live", "robotics", "computer-use")
        )
    )
    if interesting:
        warnings += 1
        print(f"\n{WARN} servable Gemini text models absent from the registry:")
        for name in interesting:
            print(f"        {name}")
        print("      These may be newer lineage points worth adding.")

    # ---- 4 and 5. live probes ---------------------------------------------
    if args.skip_probes:
        print("\n[4] Live probes skipped by request")
    else:
        print("\n[4] Generation and tool-calling probes (a fraction of one cent)")
        print(f"\n     {'model':<28}{'generate':<24}{'tool call':<28}")
        print("     " + "-" * 80)
        for model_id in configured:
            if model_id in missing:
                print(f"     {model_id:<28}{'skipped, not listed':<24}{'skipped':<28}")
                continue
            gen_ok, gen_msg = _probe_generate(client, model_id)
            fn_ok, fn_msg = _probe_function_calling(client, model_id)
            if not (gen_ok and fn_ok):
                failures += 1
            gen_cell = ("ok, " + gen_msg) if gen_ok else ("FAIL " + gen_msg)
            fn_cell = ("ok, " + fn_msg) if fn_ok else ("FAIL " + fn_msg)
            print(f"     {model_id:<28}{gen_cell:<24}{fn_cell:<28}")

    # ---- 6. AgentDojo ------------------------------------------------------
    print("\n[5] AgentDojo")
    try:
        from importlib.metadata import version as pkg_version

        from agentdojo.task_suite.load_suites import get_suite

        config = freeze.load_harness_config()
        benchmark_version = config["harness"]["benchmark_version"]
        expected = config["harness"]["agentdojo_version"]
        installed = pkg_version("agentdojo")
        if installed != expected:
            failures += 1
            print(f"{FAIL} agentdojo {installed} installed, {expected} pinned in harness.yaml")
        else:
            print(f"{OK} agentdojo {installed} matches the pinned version")

        total_user = total_inj = total_pairs = 0
        for suite_name in config["harness"]["suites"]:
            suite = get_suite(benchmark_version, suite_name)
            u, i = len(suite.user_tasks), len(suite.injection_tasks)
            total_user += u
            total_inj += i
            total_pairs += u * i
        print(
            f"{OK} {benchmark_version}: {total_user} user tasks, {total_inj} injection tasks, "
            f"{total_pairs} security cases"
        )
        print(f"      {total_pairs + total_user + total_inj} task runs per repetition per model")
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print(f"{FAIL} AgentDojo check failed: {_short_error(exc)}")

    # ---- 7. configuration freeze ------------------------------------------
    print("\n[6] Harness configuration freeze")
    try:
        matches, current, locked = freeze.check()
        if locked is None:
            warnings += 1
            print(f"{WARN} not frozen yet. Current hash would be {current}.")
            print("      Run `python src/freeze.py write` before the first sweep.")
        elif matches:
            print(f"{OK} frozen and unchanged, config_hash = {current}")
        else:
            failures += 1
            print(f"{FAIL} configuration changed since freezing: {locked} -> {current}")
            print("      Run `python src/freeze.py check` for the field-level difference.")
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print(f"{FAIL} freeze check failed: {_short_error(exc)}")

    # ---- verdict -----------------------------------------------------------
    print("\n" + "=" * 78)
    if failures:
        print(f"NOT READY: {failures} check(s) failed, {warnings} warning(s).")
        print("Resolve the failures above before spending on a sweep.")
        return 1
    print(f"READY. {warnings} warning(s), no failures.")
    print("\nNext: a smoke run for a few cents,")
    print("  python src/run_sweep.py --preset smoke --suites slack --max-user-tasks 2 --reps 1")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
