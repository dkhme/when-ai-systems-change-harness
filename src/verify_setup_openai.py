# SPDX-License-Identifier: Apache-2.0
"""Pre-flight for the OpenAI arm. Run before spending.

The house rule is that a model registry is never trusted on its own, for a
registry records what was true when somebody wrote it and a run bills against
what is true now. This checks every identifier against the live API, asserts
both wire constraints the arm depends on, and proves that function calling and
usage accounting work on the endpoint the arm actually uses.

    python src/verify_setup_openai.py

Costs a few cents. Exits non-zero on any failure.
"""

from __future__ import annotations

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

import openai  # noqa: E402

import adapters_openai as arm  # noqa: E402

PROBE_TOOL = {
    "type": "function",
    "name": "get_balance",
    "description": "Get the current account balance.",
    "parameters": {"type": "object", "properties": {}, "required": []},
    "strict": False,
}

_failures: list[str] = []


def _mark(ok: bool, label: str, detail: str = "") -> bool:
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        _failures.append(label)
    return ok


def _short(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc)[:160]}"


def main() -> int:
    registry = arm.load_registry()
    model_ids = list(registry["models"])

    print(f"registry     {arm.REGISTRY_PATH.name}, verified_on {registry['verified_on']}")
    print(f"endpoint     {arm.base_url()}")
    print(f"models       {len(model_ids)}: {', '.join(model_ids)}\n")

    if not os.environ.get("OPENAI_API_KEY"):
        print("  [FAIL] OPENAI_API_KEY is not set. Set it in the environment or in .env at the "
              "repository root, which is gitignored.")
        return 1
    client = arm.build_client()

    print("availability")
    try:
        served = {m.id for m in client.models.list()}
    except Exception as exc:  # noqa: BLE001
        _mark(False, "GET /v1/models", _short(exc))
        return 1
    for model_id in model_ids:
        _mark(model_id in served, f"{model_id} served")
    for model_id, spec in registry["models"].items():
        snapshot = spec.get("snapshot")
        if snapshot:
            _mark(
                snapshot in served,
                f"{model_id} snapshot {snapshot} served",
                "pin this rather than the floating alias",
            )

    print("\npricing present")
    for model_id, spec in registry["models"].items():
        ok = isinstance(spec.get("input"), (int, float)) and isinstance(spec.get("output"), (int, float))
        _mark(ok, f"{model_id} priced", f"{spec.get('input')}/{spec.get('output')} per MTok")

    # The load-bearing assertion is that every model accepts the temperature the
    # ARM RUNS AT, which is 1.0. Whether a model also accepts 0.0 is a recorded
    # fact about the vendor, not a pass or a fail.
    #
    # This check originally asserted that every model REJECTS 0.0, which was true
    # of gpt-5.5 and the whole 5.6 family and became false the moment Episode B
    # added gpt-5.4 and gpt-5.4-nano. Those two accept 0.0. That does not license
    # running them greedily: their successors under the three lineages are
    # gpt-5.6-terra and gpt-5.6-luna, which accept only 1.0, and comparing a
    # predecessor at 0.0 against a successor at 1.0 would vary sampling and
    # version together and confound the pair. The whole arm runs at 1.0.
    print("\nwire constraint 1: every model must accept the run temperature of 1.0")
    for model_id in model_ids:
        try:
            client.responses.create(
                model=model_id, input=[{"role": "user", "content": "ok"}], temperature=1.0, store=False
            )
            _mark(True, f"{model_id} accepts temperature=1.0")
        except Exception as exc:  # noqa: BLE001
            _mark(False, f"{model_id} accepts temperature=1.0", _short(exc))

    print("\n  recorded, not asserted: which models still permit temperature=0")
    print("  A vendor withdrawing this control removes the ability to hold sampling")
    print("  fixed across a version boundary. Where any member of a lineage refuses")
    print("  0.0, the whole lineage must run at 1.0 and the repetition floor becomes")
    print("  mandatory, because one repetition then measures one draw.")
    greedy_ok = []
    for model_id in model_ids:
        try:
            client.responses.create(
                model=model_id, input=[{"role": "user", "content": "ok"}], temperature=0.0, store=False
            )
            greedy_ok.append(model_id)
            print(f"  [--] {model_id} accepts temperature=0 (could run greedily on its own)")
        except openai.BadRequestError:
            print(f"  [--] {model_id} REJECTS temperature=0")
        except Exception as exc:  # noqa: BLE001
            _mark(False, f"{model_id} temperature=0 probe", _short(exc))
    if greedy_ok:
        print(f"  NOTE: {', '.join(greedy_ok)} accept 0.0 but are run at 1.0 anyway,")
        print("  because their successors in this registry accept only 1.0.")

    print("\nwire constraint 2: function tools work on /v1/responses at as-deployed reasoning")
    print("  (this is the whole reason the arm does not use the package's OpenAILLM)")
    for model_id in model_ids:
        try:
            resp = client.responses.create(
                model=model_id,
                input=[{"role": "user", "content": "What is my balance? Use the tool."}],
                tools=[PROBE_TOOL],
                tool_choice="auto",
                store=False,
            )
            kinds = [getattr(o, "type", None) for o in resp.output]
            _mark("function_call" in kinds, f"{model_id} calls tools", f"output items {kinds}")

            usage = resp.usage
            out_details = getattr(usage, "output_tokens_details", None)
            in_details = getattr(usage, "input_tokens_details", None)
            _mark(
                out_details is not None and hasattr(out_details, "reasoning_tokens"),
                f"{model_id} reports reasoning_tokens",
                "required, or reasoning bills silently as output",
            )
            _mark(
                in_details is not None and hasattr(in_details, "cached_tokens"),
                f"{model_id} reports cached_tokens",
                "required to show the gap between ceiling and invoice",
            )
        except Exception as exc:  # noqa: BLE001
            _mark(False, f"{model_id} calls tools", _short(exc))

    print("\nusage accounting: reasoning must not be double counted")
    # costs.py charges output_tokens + thought_tokens, so the adapter has to
    # split a total the API reports as inclusive. Assert the split here rather
    # than discover it in a bill.
    arm.install(thinking_mode="as_deployed")
    adapters = __import__("adapters")
    adapters.reset_global_usage()
    adapters.start_task_usage()
    llm = arm.build_llm(model_ids[-1], client, temperature=arm.DEFAULT_TEMPERATURE)
    try:
        # The prompt has to be hard enough to actually spend reasoning tokens.
        # A trivial one returns reasoning=0, and then the equality below holds
        # whether or not the adapter subtracts, which is a check that cannot
        # fail and therefore cannot pass either.
        resp = arm._responses_request(
            client,
            llm.model,
            [
                {
                    "role": "user",
                    "content": (
                        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more "
                        "than the ball. Work out the cost of the ball, then state only "
                        "the number."
                    ),
                }
            ],
            [],
            None,
        )
        arm._record_usage(llm.model, resp)
        totals = adapters.global_usage()
        reported = int(resp.usage.output_tokens)
        reasoning = int(getattr(resp.usage.output_tokens_details, "reasoning_tokens", 0) or 0)
        ledger = totals.output_tokens + totals.thought_tokens
        _mark(
            reasoning > 0,
            "probe actually spent reasoning tokens",
            f"{reasoning} tokens; without this the equality below is vacuous",
        )
        _mark(
            ledger == reported,
            "ledger output + thought equals the API's output_tokens",
            f"api={reported} (of which {reasoning} reasoning), ledger={ledger}",
        )
        _mark(
            totals.thought_tokens == reasoning and totals.output_tokens == reported - reasoning,
            "the split is on the right side of the line",
            f"ledger output={totals.output_tokens}, thought={totals.thought_tokens}",
        )
    except Exception as exc:  # noqa: BLE001
        _mark(False, "usage accounting", _short(exc))

    print("\nattack payload: the injected bytes must match the other arms")
    from agentdojo.attacks.base_attacks import DEFAULT_MODEL_NAME
    from agentdojo.models import MODEL_NAMES

    arm.register_model_names(model_ids)
    for model_id in model_ids:
        _mark(
            MODEL_NAMES.get(model_id) == DEFAULT_MODEL_NAME,
            f"{model_id} registered as {DEFAULT_MODEL_NAME!r}",
        )

    print()
    if _failures:
        print(f"{len(_failures)} check(s) failed:")
        for f in _failures:
            print(f"  - {f}")
        print("\nDo not spend until these are resolved.")
        return 1
    print("All checks passed. The arm is safe to run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
