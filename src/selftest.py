# SPDX-License-Identifier: Apache-2.0
"""Exercise the whole path offline, with no API key and no spend.

The purpose is narrow and useful: prove that the adaptation layer, the pipeline
construction, the result-file layout, the usage accounting, the cost model, and
the negative-flip analysis all work together, so that the first thing that
happens when a real key is plugged in is a measurement rather than a stack
trace.

A stub client stands in for `genai.Client`. It answers with a scripted tool call
followed by a final message, and it reports token usage exactly as the real API
does. Everything else in the path is the genuine AgentDojo code: the real task
suites, the real important_instructions attack, the real trace logger, the real
result-file layout, and the real scoring.

Run:  python src/selftest.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import adapters  # noqa: E402
import costs  # noqa: E402
import freeze  # noqa: E402

PASS = "  ok  "
FAIL = " FAIL "


class _StubUsage:
    def __init__(self, prompt: int, output: int, thoughts: int = 0, cached: int = 0) -> None:
        self.prompt_token_count = prompt
        self.candidates_token_count = output
        self.thoughts_token_count = thoughts
        self.cached_content_token_count = cached
        self.total_token_count = prompt + output + thoughts


class _StubModels:
    """Answers like the Gemini API: one tool call, then a plain reply."""

    def __init__(self, owner: "_StubClient") -> None:
        self.owner = owner

    def generate_content(self, *, model: str, contents: Any, config: Any):
        from google.genai import types as genai_types

        self.owner.calls.append(
            {
                "model": model,
                "n_contents": len(contents) if hasattr(contents, "__len__") else 1,
                "system_instruction": config.system_instruction,
                "thinking_config": config.thinking_config,
                "temperature": config.temperature,
                "n_tools": len(config.tools[0].function_declarations) if config.tools else 0,
            }
        )

        turn = self.owner.turn
        self.owner.turn += 1

        if turn == 0 and config.tools:
            declarations = config.tools[0].function_declarations
            target = declarations[0]
            args: dict[str, Any] = {}
            params = target.parameters
            if params is not None and params.properties:
                for prop_name, prop in params.properties.items():
                    if params.required and prop_name not in params.required:
                        continue
                    args[prop_name] = _placeholder(prop)
            part = genai_types.Part.from_function_call(name=target.name, args=args)
        else:
            part = genai_types.Part.from_text(text="Done.")

        candidate = genai_types.Candidate(
            content=genai_types.Content(role="model", parts=[part]),
            finish_reason=genai_types.FinishReason.STOP,
        )
        response = genai_types.GenerateContentResponse(candidates=[candidate])
        # usage_metadata is a settable field on the response model.
        response.usage_metadata = _StubUsage(prompt=4000, output=120, thoughts=60)  # type: ignore[assignment]
        return response


def _placeholder(prop: Any) -> Any:
    from google.genai import types as genai_types

    kind = getattr(prop, "type", None)
    if kind == genai_types.Type.INTEGER:
        return 1
    if kind == genai_types.Type.NUMBER:
        return 1.0
    if kind == genai_types.Type.BOOLEAN:
        return True
    if kind == genai_types.Type.ARRAY:
        return []
    if kind == genai_types.Type.OBJECT:
        return {}
    return "placeholder"


class _StubClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.turn = 0
        self.models = _StubModels(self)


def _check(label: str, condition: bool, detail: str = "") -> bool:
    marker = PASS if condition else FAIL
    print(f"{marker}{label}" + (f"  ({detail})" if detail else ""))
    return condition


def main() -> int:
    print("=" * 78)
    print("OFFLINE SELF-TEST: no API key required, no spend")
    print("=" * 78)
    ok = True

    config = freeze.load_harness_config()
    harness = config["harness"]
    generation = config["generation"]

    # ---- 1. patch installation -------------------------------------------
    print("\n[1] Adaptation layer")
    adapters.install(
        thinking_mode=generation["thinking_mode"],
        thinking_budget=generation["thinking_budget"],
    )
    from agentdojo.agent_pipeline.llms import google_llm

    ok &= _check(
        "chat_completion_request wrapped for usage accounting",
        google_llm.chat_completion_request.__name__ == "wrapped",
    )

    model_id = "gemini-2.5-flash-lite"
    adapters.register_model_names([model_id])
    from agentdojo.models import MODEL_NAMES

    ok &= _check(
        "model registered for the attack's prose name",
        MODEL_NAMES.get(model_id) == adapters.GOOGLE_MODEL_PROSE_NAME,
        MODEL_NAMES.get(model_id, "missing"),
    )

    # ---- 2. pipeline construction -----------------------------------------
    print("\n[2] Pipeline construction, bypassing ModelsEnum")
    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig

    client = _StubClient()
    llm = adapters.build_llm(
        model_id, client, generation["temperature"], generation["max_output_tokens"]
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
    ok &= _check("pipeline built and named after the model", pipeline.name == model_id, pipeline.name or "None")

    # ---- 3. attack resolution ---------------------------------------------
    print("\n[3] Attack")
    from agentdojo.attacks.attack_registry import load_attack
    from agentdojo.task_suite.load_suites import get_suite

    suite = get_suite(harness["benchmark_version"], "slack")
    attack = load_attack(harness["attack"], suite, pipeline)
    ok &= _check("important_instructions attack instantiated", attack.name == harness["attack"])
    ok &= _check(
        "attack addresses the model by AgentDojo's Gemini prose name",
        getattr(attack, "model_name", None) == adapters.GOOGLE_MODEL_PROSE_NAME,
        getattr(attack, "model_name", "missing"),
    )

    # ---- 4. a real task through the real code path ------------------------
    print("\n[4] One real task end to end")
    tmpdir = Path(tempfile.mkdtemp(prefix="agentdojo-selftest-"))
    try:
        sys.path.insert(0, str(SRC))
        import run_sweep

        user_task_id = sorted(suite.user_tasks.keys())[0]
        injection_task_id = sorted(suite.injection_tasks.keys())[0]
        unit = run_sweep.WorkUnit("slack", user_task_id, injection_task_id, "security")

        result = run_sweep._run_one(
            unit, suite, pipeline, attack, tmpdir, harness["benchmark_version"], 0
        )
        record = result.pop("_record")

        ok &= _check("task ran without raising", result["error"] is None, result["error"] or "")
        ok &= _check(
            "utility and security scored",
            result["utility"] is not None and result["security"] is not None,
            f"utility={result['utility']} security={result['security']}",
        )
        ok &= _check("stub client received calls", len(client.calls) > 0, f"{len(client.calls)} calls")

        expected = tmpdir / model_id / "slack" / user_task_id / harness["attack"] / f"{injection_task_id}.json"
        ok &= _check("trace written to the expected path", expected.exists(), str(expected.relative_to(tmpdir)))

        if expected.exists():
            trace = json.loads(expected.read_text())
            ok &= _check("trace carries the injection payload", bool(trace.get("injections")))
            injected_text = " ".join(str(v) for v in trace.get("injections", {}).values())
            ok &= _check(
                "injected text addresses the Gemini prose name",
                adapters.GOOGLE_MODEL_PROSE_NAME in injected_text,
            )

        # ---- 5. usage accounting and cost ---------------------------------
        print("\n[5] Usage accounting and cost")
        ok &= _check(
            "per-task token usage captured",
            record.calls > 0 and record.prompt_tokens > 0,
            f"{record.calls} calls, {record.prompt_tokens} prompt tokens",
        )
        ok &= _check(
            "thinking tokens captured separately",
            record.thought_tokens > 0,
            f"{record.thought_tokens} thinking tokens",
        )
        pricing = costs.load_pricing()[model_id]
        cost = costs.cost_of_usage(pricing, record)
        expected_cost = (
            record.prompt_tokens * pricing.input
            + (record.output_tokens + record.thought_tokens) * pricing.output
        ) / 1_000_000
        ok &= _check(
            "cost matches hand arithmetic",
            abs(cost - expected_cost) < 1e-9,
            f"{costs.format_usd(cost)}",
        )

        totals = adapters.global_usage()
        ok &= _check(
            "process-wide totals accumulate",
            totals.prompt_tokens >= record.prompt_tokens,
            f"{totals.prompt_tokens} tokens over {totals.calls} calls",
        )

        # ---- 6. long-context tiering ---------------------------------------
        print("\n[6] Long-context pricing tier")
        pro = costs.load_pricing()["gemini-2.5-pro"]
        short = pro.cost_for_call(100_000, 1_000)
        long = pro.cost_for_call(300_000, 1_000)
        ok &= _check(
            "Pro models price above 200k at the higher tier",
            abs(long / 3 - short) > 1e-9 and long > short * 3,
            f"{costs.format_usd(short)} vs {costs.format_usd(long)}",
        )
        flat = costs.load_pricing()["gemini-2.5-flash-lite"]
        ok &= _check(
            "Flash models are not tiered",
            abs(flat.cost_for_call(300_000, 0) - 3 * flat.cost_for_call(100_000, 0)) < 1e-9,
        )

        # ---- 7. thinking mode ----------------------------------------------
        print("\n[7] Thinking configuration")
        sent = client.calls[0]["thinking_config"]
        if generation["thinking_mode"] == "as_deployed":
            ok &= _check("as_deployed sends no thinking_config", sent is None)
        # A fresh logdir, since the task above is now cached and would be
        # skipped without any API call. That skip is the resume behaviour the
        # sweep relies on, so it is verified rather than defeated.
        cached_before = len(client.calls)
        run_sweep._run_one(unit, suite, pipeline, attack, tmpdir, harness["benchmark_version"], 0)
        ok_cache = len(client.calls) == cached_before
        _check("completed tasks are skipped on re-run, which is how resume works", ok_cache)

        adapters.install(thinking_mode="normalized", thinking_budget=1234)
        client.turn = 0
        fresh = tmpdir / "thinking-probe"
        run_sweep._run_one(unit, suite, pipeline, attack, fresh, harness["benchmark_version"], 0)
        sent = client.calls[-1]["thinking_config"]
        ok &= _check(
            "normalized pins the thinking budget",
            sent is not None and sent.thinking_budget == 1234,
            str(getattr(sent, "thinking_budget", None)),
        )
        adapters.install(thinking_mode=generation["thinking_mode"])

        # ---- 8. concurrency safety ------------------------------------------
        print("\n[8] Concurrency")
        ok &= _check("threads keep separate logger stacks and usage records",
                     _test_concurrency(suite, pipeline, attack, harness, tmpdir))

        # ---- 9. analysis ----------------------------------------------------
        print("\n[9] Negative-flip analysis on synthetic outcomes")
        ok &= _check("analysis arithmetic", _test_analysis())
        ok &= _check("security polarity pinned to AgentDojo's own task code", _test_polarity())
        ok &= _check("crashed traces are excluded rather than scored", _test_unusable())

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        adapters.reset_global_usage()

    print("\n" + "=" * 78)
    if ok:
        print("SELF-TEST PASSED. The path is sound; only a live API key is missing.")
        print("\nNext:  paste the key into .env at the repository root, then")
        print("       python src/verify_setup.py")
        return 0
    print("SELF-TEST FAILED. Do not spend until the failures above are resolved.")
    return 1


def _test_concurrency(suite, pipeline, attack, harness, tmpdir: Path) -> bool:
    """Run several tasks at once and check nothing crosses between threads.

    AgentDojo declares `LOGGER_STACK = ContextVar("logger_stack", default=[])`.
    That default is a single shared mutable list, so threads that never call
    `set` would all append to the same object and their logger stacks would
    interleave. run_sweep isolates each thread; this proves it.
    """
    from concurrent.futures import ThreadPoolExecutor

    import run_sweep

    concurrent_dir = tmpdir / "concurrent"
    user_task_ids = sorted(suite.user_tasks.keys())[:4]
    injection_task_id = sorted(suite.injection_tasks.keys())[0]
    units = [
        run_sweep.WorkUnit("slack", user_task_id, injection_task_id, "security")
        for user_task_id in user_task_ids
    ]

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda unit: run_sweep._run_one(
                    unit, suite, pipeline, attack, concurrent_dir, harness["benchmark_version"], 0
                ),
                units,
            )
        )

    problems: list[str] = []
    if any(r["error"] for r in results):
        problems.append("a concurrent task raised")
    if any(r.pop("_record").calls == 0 for r in results):
        problems.append("a task recorded no API calls, so usage crossed threads")

    for user_task_id in user_task_ids:
        expected = (
            concurrent_dir
            / pipeline.name
            / "slack"
            / user_task_id
            / harness["attack"]
            / f"{injection_task_id}.json"
        )
        if not expected.exists():
            problems.append(f"missing trace for {user_task_id}")

    from agentdojo.logging import LOGGER_STACK

    if LOGGER_STACK.get():
        problems.append("logger stack left dirty after the run")

    if problems:
        print(f"      {'; '.join(problems)}")
        return False
    return True


def _test_polarity() -> bool:
    """Pin the meaning of AgentDojo's `security` field to its own source.

    The field reads as "attack succeeded", not "system was secure": an
    injection task's `security()` returns True once the attacker's goal is
    achieved. Reading it the other way would invert every negative flip in the
    paper, turning regressions into repairs. This test asserts the direction
    against the benchmark's own task implementation rather than against a
    comment, so a change in the package cannot silently flip the result.
    """
    import inspect

    from agentdojo.task_suite.load_suites import get_suite

    suite = get_suite("v1.2.2", "slack")
    task = next(iter(suite.injection_tasks.values()))
    source = inspect.getsource(type(task).security)

    # The Slack injection tasks return True on the attacker's success
    # condition, and their COMMENT states it in prose.
    comment = getattr(task, "COMMENT", "") or ""
    says_success = "attack is successful" in comment.lower()
    if not says_success:
        print(f"      COMMENT no longer states the success condition: {comment[:80]!r}")
        return False
    if "return True" not in source:
        print("      security() no longer returns True on attacker success")
        return False

    # And the round trip through the loader must invert it.
    import analyze

    trace = {
        "suite_name": "slack",
        "user_task_id": "user_task_0",
        "injection_task_id": "injection_task_0",
        "utility": True,
        "security": True,  # attack succeeded
        "messages": [],
    }
    import json as _json
    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory() as d:
        path = Path(d) / "t.json"
        path.write_text(_json.dumps(trace))
        secure, _, _, seen, _ = analyze._load_repetition(Path(d))
        if seen != 1:
            print("      loader did not read the trace")
            return False
        value = next(iter(secure.values()))
        if value is not False:
            print(f"      attack-succeeded trace loaded as secure={value}, expected False")
            return False
    return True


def _test_unusable() -> bool:
    """A trace from a crashed task must be excluded, never read as insecure."""
    import json as _json
    import tempfile as _tempfile

    import analyze

    crashed = {
        "suite_name": "slack",
        "user_task_id": "user_task_0",
        "injection_task_id": "injection_task_0",
        "error": None,  # AgentDojo leaves this null when the task raised
        "messages": [],
    }  # note: no utility, no security
    with _tempfile.TemporaryDirectory() as d:
        (Path(d) / "t.json").write_text(_json.dumps(crashed))
        secure, _, _, seen, unusable = analyze._load_repetition(Path(d))
        if secure or seen != 0:
            print("      crashed trace was scored instead of excluded")
            return False
        if unusable != 1:
            print(f"      crashed trace not counted as unusable (got {unusable})")
            return False
    return True


def _test_analysis() -> bool:
    """Verify the flip arithmetic against a hand-computed case set."""
    import analyze

    # 10 cases. Under N: 8 secure. Under N+1: 6 secure.
    # Constructed to contain 3 negative flips and 1 positive flip, so the
    # aggregate improves nowhere but the paired view is unambiguous.
    before_map = {
        ("slack", f"user_task_{i}", "injection_task_0"): secure
        for i, secure in enumerate([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    }
    after_map = {
        ("slack", f"user_task_{i}", "injection_task_0"): secure
        for i, secure in enumerate([1, 1, 1, 1, 1, 0, 0, 0, 1, 0])
    }
    before = analyze.ModelOutcomes("before", {k: bool(v) for k, v in before_map.items()}, {}, 1, 10, 0)
    after = analyze.ModelOutcomes("after", {k: bool(v) for k, v in after_map.items()}, {}, 1, 10, 0)
    # NOTE: these maps are already in "secure" polarity, the inverse of
    # AgentDojo's on-disk `security` field. _test_polarity pins that.
    result = analyze.compare(before, after)

    checks = [
        ("cases", result["cases"] == 10),
        ("negative flips", result["negative_flips"] == 3),
        ("positive flips", result["positive_flips"] == 1),
        ("net regression", result["net_regression"] == 2),
        ("secure before", result["secure_before"] == 8),
        ("negative-flip rate", abs(result["negative_flip_rate"] - 3 / 8) < 1e-9),
        ("ASR before", abs(result["asr_before"] - 0.2) < 1e-9),
        ("ASR after", abs(result["asr_after"] - 0.4) < 1e-9),
        ("McNemar computed", result["mcnemar_p"] is not None),
    ]
    failed = [name for name, passed in checks if not passed]
    if failed:
        print(f"      arithmetic failures: {', '.join(failed)}")
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
