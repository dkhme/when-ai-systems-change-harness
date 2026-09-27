# SPDX-License-Identifier: Apache-2.0
"""Adapt AgentDojo 0.1.35 to the Anthropic Messages API.

Mirrors the function surface of `adapters.py` and `adapters_baseten.py` so
`run_sweep.py --arm anthropic` binds this module and changes nothing else.

AgentDojo ships a native `AnthropicLLM`, so unlike the Baseten arm there is no
client to shim. Four things still need handling, and the first is the one that
could have silently manufactured a result.

**1. `max_tokens` defaults to 1024, and it is REQUIRED by this API.**
`AnthropicLLM._MAX_TOKENS = 1024`. The Google and OpenAI paths send no limit at
all, so on those arms the model's own cap applies and never binds. Here a
1024-token cap would truncate long agent turns. A truncated turn emits no valid
tool call, fails the user task, and — because a model that emits nothing also
injects nothing — **scores as SECURE**. That is the degeneracy confound that
`degeneracy_check.py` tests for, except manufactured by our own harness rather
than by the model. We raise the cap to 8192 and `verify_setup_anthropic.py` asserts that
no probe response stops on `max_tokens`; the runner records the value it sent in
every manifest.

**2. Nothing records token usage.** Same gap as the Gemini path. We wrap
`chat_completion_request` and translate Anthropic's `usage` into the shared
ledger, so the budget ceiling means the same thing on every arm.

**3. The closed `ModelsEnum` stops at `claude-3-7-sonnet`.** Every model in this
arm postdates the pinned package. We construct `AnthropicLLM` directly and pass
it as `PipelineConfig.llm`, exactly as the Gemini adapter does. The enum is
bypassed, never edited.

**4. Thinking forces temperature 1.** `AnthropicLLM.__init__` sets
`self.temperature = 1` whenever `thinking_budget_tokens` is not None, because the
API requires it. So a thinking-normalized arm here cannot also be temperature 0,
and determinism and reasoning-normalization are mutually exclusive on this
provider. This arm therefore runs `as_deployed` only, and `install()` refuses
anything else rather than silently returning a non-deterministic run.

`as_deployed` means something different here than on Gemini, and the difference
is worth stating rather than burying. Gemini applies its own server-side
reasoning default when we send no thinking configuration. The Anthropic API
enables extended thinking only when asked, so `as_deployed` on this arm means
**thinking is off**. A cross-family comparison of these two `as_deployed` arms is
comparing a model with its vendor default against a model with thinking
disabled. That is recorded in `arm_notes` and must be carried into any write-up.

One thing deliberately NOT done: prompt caching. A `cache_control` breakpoint
would cut spend substantially on a workload with this much repeated prefix, but
cache reads bill at 0.1x and every other arm prices cached tokens at the full
rate to keep cost an upper bound. Adding caching here would make this arm's cost
figures mean something different from the other two. If a future round wants it,
add it together with the pricing change, not before.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import anthropic
import yaml
from agentdojo.agent_pipeline.llms import anthropic_llm
from agentdojo.attacks.base_attacks import DEFAULT_MODEL_NAME
from agentdojo.models import MODEL_NAMES

import adapters
from adapters import note_error, note_rate_limit, record_call

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
REGISTRY_PATH = CONFIG_DIR / "models_anthropic.yaml"

REQUEST_TIMEOUT_S = 600.0
MAX_RETRIES = 3

# See point 1 in the module docstring. High enough never to bind on an AgentDojo
# turn, which is a tool call plus a short preamble, and asserted non-binding by
# the pre-flight.
MAX_OUTPUT_TOKENS = 8192

# The name-neutral string the `important_instructions_no_names` attack uses. It
# must be identical to the other arms' or the injected bytes differ and the
# cross-family comparison is void. Imported rather than re-spelled so the two
# cannot drift apart.
ANTHROPIC_MODEL_PROSE_NAME = DEFAULT_MODEL_NAME

# Models that reject the `temperature` field outright, returning
# "`temperature` is deprecated for this model". Verified live 2026-08-10:
# claude-sonnet-4-5 and claude-sonnet-4-6 accept 0.0, 1.0 and omission;
# claude-sonnet-5 rejects 0.0 and accepts only its default of 1.0.
#
# The consequence is not a wire detail, it is a measurement constraint. The
# lineage can only be compared at a temperature every member accepts, which is
# 1.0, so **this arm cannot be run greedily**. Determinism is unavailable, the
# repetition floor stops being optional, and a single-repetition pass over this
# lineage reports one draw rather than the model's behaviour.
#
# This belongs in the paper. A vendor withdrew a sampling control between two
# consecutive versions of one model tier, and in doing so removed the ability to
# hold a variable fixed across that tier. It is the same species of decay as
# AgentDojo losing the ability to run Gemini 3.x, and it argues the same point.
NO_TEMPERATURE_CONTROL = frozenset({"claude-sonnet-5"})
DEFAULT_TEMPERATURE = 1.0

_PATCHED = False
_PATCH_LOCK = threading.Lock()

# Constructor arguments for the per-call client built inside `install()`'s
# wrapper. Populated by `build_client()`, and derived lazily from the
# environment if the wrapper somehow runs first — the runner calls `install()`
# before `build_client()`, so the wrapper must not depend on that order.
_CLIENT_KWARGS: dict[str, Any] = {}


def _client_kwargs() -> dict[str, Any]:
    if not _CLIENT_KWARGS:
        build_client()
    return _CLIENT_KWARGS


def load_registry(path: Path | None = None) -> dict[str, Any]:
    """Read the model registry. Never trusted on its own; see the file header."""
    with open(path or REGISTRY_PATH) as fh:
        return yaml.safe_load(fh)


def base_url() -> str:
    """Reported in the manifest so a run records the endpoint it actually hit."""
    return os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")


def pipeline_name(model_id: str) -> str:
    """Anthropic ids carry no path separator, so the id is its own alias."""
    return model_id


def build_client(api_key: str | None = None) -> anthropic.Anthropic:
    """Build the client the pipeline holds, and record how to rebuild it.

    The returned client satisfies `AnthropicLLM`'s constructor. It is NOT the
    client that issues requests: see the per-call client in `install()` for why
    a shared one deadlocks. The kwargs are stashed so the wrapper can rebuild
    an equivalent client inside whichever event loop is running.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Set it in the environment or in .env at "
            "the repository root, which is gitignored. Never paste a key into a "
            "source file or a manifest."
        )
    kwargs: dict[str, Any] = {
        "api_key": key,
        "timeout": REQUEST_TIMEOUT_S,
        "max_retries": MAX_RETRIES,
    }
    if "ANTHROPIC_BASE_URL" in os.environ:
        kwargs["base_url"] = os.environ["ANTHROPIC_BASE_URL"]
    _CLIENT_KWARGS.clear()
    _CLIENT_KWARGS.update(kwargs)
    return anthropic.Anthropic(**kwargs)


def _is_rate_limit(exc: Exception) -> bool:
    if isinstance(exc, anthropic.RateLimitError):
        return True
    status = getattr(exc, "status_code", None)
    return status == 429


def install(thinking_mode: str = "as_deployed", thinking_budget: int = 4096) -> None:
    """Wrap the package's request function so usage reaches the shared ledger.

    Idempotent and lock-guarded: the runner is threaded and may call this once
    per worker.

    Refuses any thinking mode other than `as_deployed`. See point 4 in the
    module docstring: enabling extended thinking forces temperature 1 on this
    API, so a normalized arm would silently stop being deterministic. Failing
    loudly is the only honest option, because the resulting run would look
    perfectly ordinary and its repetition floor would be meaningless.
    """
    global _PATCHED
    if thinking_mode != "as_deployed":
        raise RuntimeError(
            f"thinking_mode={thinking_mode!r} is not available on the Anthropic arm. "
            "Extended thinking forces temperature=1 on this API "
            "(AnthropicLLM.__init__), so a normalized arm cannot also be "
            "deterministic. Run as_deployed, and note in the write-up that "
            "as_deployed here means thinking OFF, unlike Gemini where it means "
            "the vendor's own default."
        )
    with _PATCH_LOCK:
        if _PATCHED:
            return
        original = anthropic_llm.chat_completion_request

        async def _counting_request(client, model, messages, tools, **kwargs):
            # A FRESH AsyncAnthropic per call, deliberately ignoring the one the
            # pipeline holds. This is a correctness fix, not a style choice.
            #
            # `AnthropicLLM.query` calls `asyncio.run(...)` for every turn, which
            # creates and destroys a NEW event loop each time, and the runner
            # drives several worker threads. The `AsyncAnthropic` built once in
            # `AnthropicLLM.__init__` owns an `httpx.AsyncClient` whose connection
            # pool binds to whichever loop first used it. Every later call from a
            # different loop awaits a pool tied to a loop that no longer runs, and
            # a fraction of them never return.
            #
            # Observed, not theorised: a 3-model pass wedged 29 minutes into
            # model 1 with 463 of 467 tasks done, four tasks blocked, and **zero
            # ESTABLISHED sockets** — nothing in flight, so the 600 s request
            # timeout never fired either. A stall with no error is the worst
            # failure mode available: the runner reports 0 errors and simply
            # stops.
            #
            # Binding the client inside the running loop costs one TLS handshake
            # per call, on the order of 100 ms against turns that take tens of
            # seconds. Cheap insurance against a silent hang.
            try:
                async with anthropic.AsyncAnthropic(**_client_kwargs()) as fresh:
                    completion = await original(fresh, model, messages, tools, **kwargs)
            except Exception as exc:  # noqa: BLE001 - counted, then re-raised
                if _is_rate_limit(exc):
                    note_rate_limit()
                note_error()
                raise
            usage = getattr(completion, "usage", None)
            if usage is not None:
                cached = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
                created = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
                record_call(
                    model=model,
                    # Anthropic reports uncached input separately from cache
                    # reads and writes. Sum them so `prompt_tokens` means the
                    # same thing as on the other arms, and price the whole sum
                    # at the full input rate -- the upper-bound convention this
                    # study uses everywhere.
                    prompt_tokens=int(getattr(usage, "input_tokens", 0) or 0) + cached + created,
                    output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                    cached_tokens=cached,
                )
            return completion

        anthropic_llm.chat_completion_request = _counting_request
        _PATCHED = True


def register_model_names(model_ids: list[str]) -> None:
    """Teach AgentDojo's attack layer a prose name for each model.

    The package's own table maps every `claude-*` id it knows to the literal
    string "Claude", and `important_instructions` interpolates that into the
    injected message. This arm runs `important_instructions_no_names`, which
    interpolates nothing, so the table is not consulted for the attack text.

    We register anyway, against both the id and its alias, for the same reason
    the Baseten arm does: `get_model_name_from_pipeline` resolves by substring
    match and raises when nothing matches. Registering the name-neutral string
    means that if a future run is ever pointed at the naming attack by mistake,
    it produces the SAME injected bytes as the other arms rather than a
    Claude-specific payload that would silently confound the comparison.
    """
    for model_id in model_ids:
        MODEL_NAMES.setdefault(model_id, ANTHROPIC_MODEL_PROSE_NAME)
        MODEL_NAMES.setdefault(pipeline_name(model_id), ANTHROPIC_MODEL_PROSE_NAME)


def build_llm(
    model_id: str,
    client: anthropic.Anthropic,
    temperature: float = 0.0,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
) -> anthropic_llm.AnthropicLLM:
    """Construct an AnthropicLLM bound to this client.

    `thinking_budget_tokens` is left None, which keeps temperature at the value
    passed. `max_tokens` is sent because this API requires it; see point 1.

    The harness config carries 65535 for the Gemini arm, which is above Gemini's
    own cap and therefore not binding. Anthropic rejects a value above the
    model's limit, so we clamp to MAX_OUTPUT_TOKENS and the manifest records
    what was actually sent rather than what was configured.
    """
    if model_id in NO_TEMPERATURE_CONTROL and temperature != DEFAULT_TEMPERATURE:
        raise RuntimeError(
            f"{model_id} rejects the temperature field "
            f"('`temperature` is deprecated for this model'), so it cannot run at "
            f"temperature={temperature}. Its predecessors accept 0.0, but comparing "
            f"versions at DIFFERENT temperatures would vary two things at once and "
            f"confound the version with the sampling. Run the whole lineage at "
            f"--temperature {DEFAULT_TEMPERATURE} with --allow-config-change, and "
            f"treat the repetition floor as mandatory: temperature 1.0 is not "
            f"deterministic, so one repetition measures one draw."
        )
    llm = anthropic_llm.AnthropicLLM(
        client=client,
        model=model_id,
        temperature=temperature,
        thinking_budget_tokens=None,
        max_tokens=min(max_output_tokens, MAX_OUTPUT_TOKENS),
    )
    llm.name = pipeline_name(model_id)
    return llm


# The runner reads these through the module surface; re-exported so every arm
# exposes the identical set.
start_task_usage = adapters.start_task_usage
global_usage = adapters.global_usage
rate_limit_hits = adapters.rate_limit_hits
reset_global_usage = adapters.reset_global_usage
