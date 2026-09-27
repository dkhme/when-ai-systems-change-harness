# SPDX-License-Identifier: Apache-2.0
"""Runtime adaptation of AgentDojo 0.1.35 for Baseten's Model APIs.

This is the open-weight arm of the study. It presents the identical function
surface as `adapters.py` so `run_sweep.py` selects an arm by binding one module
rather than by branching through the runner.

Baseten serves open-weight models behind an OpenAI-compatible endpoint, so
AgentDojo's released `OpenAILLM` is the right pipeline element and is used
unmodified. Four gaps stand between it and this study, and each is closed here.

1.  `OpenAILLM` builds nothing; it takes a client. We build an `openai.OpenAI`
    pointed at Baseten with an explicit request timeout, for the same reason the
    Gemini adapter sets one: a stalled connection with no timeout blocks its
    worker forever, and AgentDojo's tenacity retry fires on exceptions, never on
    a silent hang.

2.  Baseten's request validator is stricter than OpenAI's, and AgentDojo emits
    two shapes it rejects. See `_sanitize_message`.

3.  Nothing records token usage, so cost would stay an estimate. We wrap
    `chat_completion_request` and read `usage` off every response, which turns
    the budget ceiling into a measurement.

4.  Served model ids carry a slash (`moonshotai/Kimi-K2.6`). AgentDojo derives
    its results cache path from the pipeline name, so an unsanitised id would
    silently split every model's traces across an extra directory level. See
    `pipeline_name`.

Token accounting is deliberately shared with `adapters.py` rather than
reimplemented, so both arms accumulate through one ledger and a dollar means the
same thing on each.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import openai
import yaml
from openai._types import NOT_GIVEN
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from agentdojo.agent_pipeline.llms import openai_llm
from agentdojo.attacks.base_attacks import DEFAULT_MODEL_NAME
from agentdojo.models import MODEL_NAMES

# The shared token ledger. Re-exported so run_sweep.py can bind either adapter
# module and call the same names.
from adapters import (  # noqa: F401
    UsageRecord,
    global_usage,
    note_error,
    note_rate_limit,
    rate_limit_hits,
    record_call,
    reset_global_usage,
    start_task_usage,
)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
REGISTRY_PATH = CONFIG_DIR / "models_baseten.yaml"

DEFAULT_BASE_URL = "https://inference.baseten.co/v1"

# Per-request timeout in seconds. Ten minutes, matching the Gemini adapter.
REQUEST_TIMEOUT_S = 600.0

# The prose name registered for every model on this arm.
#
# AgentDojo's `important_instructions` attack interpolates the target model's
# name into the injected message, resolved through MODEL_NAMES, and
# `get_model_name_from_pipeline` raises for an unregistered id. That creates a
# cross-family confound: a Claude backbone is addressed as "Claude" and a Gemini
# backbone as "AI model developed by Google", so a difference in attack success
# would confound the backbone with the wording of the attack.
#
# We register AgentDojo's OWN default victim name, which is the string the
# `important_instructions_no_names` variant substitutes anyway. Two consequences,
# both wanted. Under `no_names`, the arm this study runs, the value here is
# overwritten by the attack and provably cannot affect the payload. Under plain
# `important_instructions` it is used, and produces the same name-neutral text,
# so the two attacks are byte-identical on this arm and neither smuggles a
# family-specific string into the injection.
OPEN_WEIGHT_MODEL_PROSE_NAME = DEFAULT_MODEL_NAME


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def load_registry(path: Path | None = None) -> dict[str, Any]:
    with (path or REGISTRY_PATH).open() as f:
        return yaml.safe_load(f)


def base_url() -> str:
    """Endpoint, from the environment if set, otherwise from the registry."""
    from_env = os.environ.get("BASETEN_BASE_URL")
    if from_env:
        return from_env.rstrip("/")
    return str(load_registry().get("base_url", DEFAULT_BASE_URL)).rstrip("/")


def pipeline_name(model_id: str) -> str:
    """Filesystem-safe alias for a served model id.

    `AgentPipeline.from_config` takes the pipeline name from `llm.name`, and
    AgentDojo caches results at
    `logdir/<pipeline>/<suite>/<user_task>/<attack>/<injection_task>.json`. A
    served id such as `moonshotai/Kimi-K2.6` would therefore introduce a
    directory boundary inside the pipeline segment, and `analyze.py` addresses a
    model by exactly one directory under the run root. Substituting the
    separator keeps every path one segment deep while the true id, which is what
    goes on the wire, stays on `llm.model`.
    """
    return model_id.replace("/", "__")


# --------------------------------------------------------------------------
# Request shaping
# --------------------------------------------------------------------------


def _sanitize_message(message: Any) -> Any:
    """Rewrite one OpenAI-shaped message into what Baseten's validator accepts.

    Two rewrites, both of shape rather than of content.

    `role="developer"` becomes `role="system"`. AgentDojo emits the system
    message under OpenAI's newer developer role, which Baseten rejects. The two
    carry the same text to the same place in the conversation, and the frozen
    configuration hashes the resolved system message TEXT, so this does not move
    the hash and does not change what the model is told.

    A list of content parts becomes a plain string. AgentDojo emits
    `[{"type": "text", "text": ...}, ...]`; concatenating the parts in order
    reproduces the same characters, which `selftest_baseten.py` asserts rather
    than assumes.

    Both were found by running the benchmark: the same rewrites were needed for
    the AgentDojo banking cell run against this endpoint on 2 July 2026.
    """
    message = dict(message)
    if message.get("role") == "developer":
        message["role"] = "system"
    content = message.get("content")
    if isinstance(content, list):
        message["content"] = "".join(
            (part.get("text", "") if isinstance(part, dict) else str(part)) for part in content
        )
    return message


class _ShimCompletions:
    """Sanitises outgoing messages and backs off on rate limits."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def create(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("messages"):
            kwargs["messages"] = [_sanitize_message(m) for m in kwargs["messages"]]
        for attempt in range(6):
            try:
                return self._inner.create(*args, **kwargs)
            except openai.RateLimitError:
                note_rate_limit()
                if attempt == 5:
                    raise
                time.sleep(min(60, 5 * (2**attempt)))
        raise RuntimeError("unreachable")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _ShimChat:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    @property
    def completions(self) -> _ShimCompletions:
        return _ShimCompletions(self._inner.completions)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class ShimmedBasetenClient:
    """An OpenAI client whose chat completions are shaped for Baseten.

    Everything other than `chat.completions.create` delegates through untouched,
    so `models.list()` and the rest of the SDK behave normally.
    """

    def __init__(self, inner: openai.OpenAI) -> None:
        self._inner = inner

    @property
    def chat(self) -> _ShimChat:
        return _ShimChat(self._inner.chat)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


# --------------------------------------------------------------------------
# Client construction
# --------------------------------------------------------------------------


def build_client(api_key: str | None = None) -> ShimmedBasetenClient:
    """Build a Baseten Model APIs client from an API key.

    The key comes from `api_key` or, failing that, from BASETEN_API_KEY in the
    environment, which `run_sweep.py` and the pre-flight scripts also populate
    from .env at the repository root. A short-lived key, revoked after the run,
    is recommended; the value `<revoked>` is treated as no key at all.
    """
    key = api_key or os.environ.get("BASETEN_API_KEY")
    if not key or key == "<revoked>":
        raise RuntimeError(
            "No Baseten API key found. Set BASETEN_API_KEY in the environment or "
            "in .env at the repository root."
        )
    inner = openai.OpenAI(
        api_key=key,
        base_url=base_url(),
        timeout=REQUEST_TIMEOUT_S,
    )
    return ShimmedBasetenClient(inner)


# --------------------------------------------------------------------------
# Patching
# --------------------------------------------------------------------------

_PATCHED = False


def install(thinking_mode: str = "as_deployed", thinking_budget: int = 4096) -> None:
    """Install usage accounting on the OpenAI-compatible call path.

    Idempotent. The signature matches `adapters.install` so the runner can bind
    either module, but the thinking arguments have no effect here: Baseten's
    Model APIs expose no reasoning-budget control, and `OpenAILLM` sends
    `reasoning_effort` only when one is set, which it is not. `as_deployed` is
    therefore the only mode this arm can occupy, and it is the mode the frozen
    configuration specifies. Any other value is refused rather than silently
    ignored, because silently ignoring it would let a run claim a normalized
    condition it did not have.
    """
    global _PATCHED
    if thinking_mode != "as_deployed":
        raise ValueError(
            f"thinking_mode={thinking_mode!r} cannot be honoured on the Baseten arm: "
            "the endpoint exposes no reasoning-budget control. Only 'as_deployed' "
            "is available here."
        )
    if _PATCHED:
        return

    def wrapped(client, model, messages, tools, reasoning_effort, temperature=0.0):
        try:
            response = _corrected_request(
                client, model, messages, tools, reasoning_effort, temperature
            )
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
            # `_ShimCompletions.create` already counts every RateLimitError it
            # sees, including the ones it retries past, so counting again here
            # would inflate the terminal failure to two. Anything else that
            # looks like a 429 reached us without passing through that loop and
            # is counted here.
            if _is_rate_limit(exc) and not isinstance(exc, openai.RateLimitError):
                note_rate_limit()
            note_error()
            raise

        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            record_call(
                model,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                # Reasoning tokens, where a model reports them, are already
                # inside completion_tokens on an OpenAI-compatible response and
                # bill at the output rate, so counting them again would double.
                # They are recorded separately only for visibility.
                thought_tokens=0,
                cached_tokens=_cached_tokens(usage),
            )
        return response

    openai_llm.chat_completion_request = wrapped
    _PATCHED = True


@retry(
    # Identical to the settings the released chat_completion_request carries, so
    # replacing it changes the temperature bug below and nothing else.
    wait=wait_random_exponential(multiplier=1, max=40),
    stop=stop_after_attempt(3),
    reraise=True,
    retry=retry_if_not_exception_type((openai.BadRequestError, openai.UnprocessableEntityError)),
)
def _corrected_request(client, model, messages, tools, reasoning_effort, temperature):
    """AgentDojo 0.1.35's OpenAI request, with the falsy-zero temperature fixed.

    THE DEFECT. The published 0.1.35 wheel writes

        temperature=temperature or NOT_GIVEN

    and `0.0 or NOT_GIVEN` is `NOT_GIVEN`, so a configured temperature of ZERO
    is the one value that never reaches the wire. The endpoint then samples at
    its own default. The frozen configuration specifies temperature 0.0 and the
    configuration hash certifies it, but the hash covers the config file rather
    than the request, so nothing downstream would have caught this: the arm
    would have run at an undeclared sampling temperature while every manifest
    reported 0.0.

    SCOPE, checked rather than assumed. Only the OpenAI-compatible path is
    affected. `google_llm.py` passes `temperature=self.temperature` straight into
    the generation config with no falsy coercion, so the completed Gemini arm did
    run at temperature 0. `anthropic_llm.py` already writes the `is not None`
    form. The same-version source checkout also carries the `is not None` form,
    so the published wheel and the repository disagree at one version number,
    which is worth a sentence in the paper for the same reason the
    thought-signature incompatibility is.

    WHY CORRECT IT HERE rather than upgrade. `agentdojo_version` is inside the
    frozen fingerprint, so moving the pin would invalidate the Gemini arm and
    every result already recorded. Correcting the call in the adapter is the same
    remedy `adapters.py` applies to Gemini's thought signatures, and it leaves
    the benchmark installed as a pinned, citable dependency.
    """
    return client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools or NOT_GIVEN,
        tool_choice="auto" if tools else NOT_GIVEN,
        temperature=temperature if temperature is not None else NOT_GIVEN,
        reasoning_effort=reasoning_effort or NOT_GIVEN,
    )


def _cached_tokens(usage: Any) -> int:
    """Cached prompt tokens, if the provider reports them.

    Recorded, then billed at the full input rate exactly as the Gemini arm does,
    so every cost figure stays an upper bound.
    """
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return 0
    return int(getattr(details, "cached_tokens", 0) or 0)


def _is_rate_limit(exc: Exception) -> bool:
    if isinstance(exc, openai.RateLimitError):
        return True
    status = getattr(exc, "status_code", None)
    if status in (429, "429"):
        return True
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "too many requests" in text


def register_model_names(model_ids: list[str]) -> None:
    """Teach AgentDojo's attack layer a prose name for each model.

    Registered against BOTH the served id and its pipeline alias, since
    `get_model_name_from_pipeline` resolves the pipeline's name by substring
    match against MODEL_NAMES and the pipeline carries the alias.

    Every entry maps to the same name-neutral string. See
    OPEN_WEIGHT_MODEL_PROSE_NAME for why that specific string.
    """
    for model_id in model_ids:
        MODEL_NAMES.setdefault(model_id, OPEN_WEIGHT_MODEL_PROSE_NAME)
        MODEL_NAMES.setdefault(pipeline_name(model_id), OPEN_WEIGHT_MODEL_PROSE_NAME)


def build_llm(
    model_id: str,
    client: ShimmedBasetenClient,
    temperature: float = 0.0,
    max_output_tokens: int = 65535,
) -> openai_llm.OpenAILLM:
    """Construct an OpenAILLM bound to the Baseten client.

    `max_output_tokens` is accepted for signature parity with the Gemini adapter
    and is NOT sent: AgentDojo's OpenAI path exposes no max-tokens argument, so
    each model applies its own default. On the Gemini arm the configured 65535
    is above Gemini's own cap and so is likewise not binding. The difference is
    recorded in the run manifest rather than left to be discovered later.
    """
    llm = openai_llm.OpenAILLM(
        client=client,  # type: ignore[arg-type]  - duck-typed shim, not an openai.OpenAI
        model=model_id,
        temperature=temperature,
    )
    llm.name = pipeline_name(model_id)
    return llm
