# SPDX-License-Identifier: Apache-2.0
"""Runtime adaptation of AgentDojo 0.1.35 for the Gemini Developer API.

AgentDojo is used as an unmodified pinned dependency rather than a fork, so the
benchmark itself stays citable and reproducible. Three gaps stand between the
released package and this study, and each is closed here at import time.

1.  The `google` provider hard-wires a Vertex AI client:

        genai.Client(vertexai=True, project=..., location=...)

    which requires a GCP project and `gcloud auth application-default login`.
    We build a Developer API client from `GEMINI_API_KEY` instead.

2.  `ModelsEnum` is a closed enum whose newest Google entries are the withdrawn
    preview identifiers `gemini-2.5-pro-preview-05-06` and
    `gemini-2.5-flash-preview-04-17`. Rather than edit the enum, we construct
    `GoogleLLM` directly and pass the instance as `PipelineConfig.llm`, which
    the released package already accepts. The enum is bypassed entirely.

3.  Nothing records token usage, so cost stays an estimate. We wrap
    `chat_completion_request` to accumulate `usage_metadata` per task, which
    turns the projected budget into a measured one.

The wrapper also carries the thinking configuration, for a model's reasoning
default is a confound whenever versions are compared.
"""

from __future__ import annotations

import base64
import contextvars
import os
import threading
from dataclasses import dataclass, field
from typing import Any

from google import genai
from google.genai import types as genai_types

from agentdojo.agent_pipeline.llms import google_llm
from agentdojo.models import MODEL_NAMES

# Every Gemini entry already in AgentDojo maps to this exact string, and the
# important_instructions attack interpolates it into the injected text. Reusing
# it verbatim keeps the injected payload byte-identical across every Gemini
# model, which is what makes the within-family comparison valid.
GOOGLE_MODEL_PROSE_NAME = "AI model developed by Google"

# Per-request timeout in milliseconds. See build_client for why this exists.
REQUEST_TIMEOUT_MS = 600_000


# --------------------------------------------------------------------------
# Usage accounting
# --------------------------------------------------------------------------


@dataclass
class UsageRecord:
    """Token counts for one task, summed over every API call the task made."""

    model: str = ""
    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    cached_tokens: int = 0
    # Highest single-call prompt size, used to detect crossing the long-context
    # pricing threshold on the Pro models.
    max_prompt_tokens: int = 0
    errors: int = 0
    per_call_prompt_tokens: list[int] = field(default_factory=list)

    def add(self, model: str, usage: Any) -> None:
        self.model = model
        self.calls += 1
        prompt = int(getattr(usage, "prompt_token_count", 0) or 0)
        # candidates_token_count excludes thinking tokens; both bill as output.
        output = int(getattr(usage, "candidates_token_count", 0) or 0)
        thoughts = int(getattr(usage, "thoughts_token_count", 0) or 0)
        cached = int(getattr(usage, "cached_content_token_count", 0) or 0)

        self.prompt_tokens += prompt
        self.output_tokens += output
        self.thought_tokens += thoughts
        self.cached_tokens += cached
        self.max_prompt_tokens = max(self.max_prompt_tokens, prompt)
        self.per_call_prompt_tokens.append(prompt)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "thought_tokens": self.thought_tokens,
            "cached_tokens": self.cached_tokens,
            "max_prompt_tokens": self.max_prompt_tokens,
            "errors": self.errors,
        }


# One record per task, isolated per thread by contextvars.
_CURRENT_USAGE: contextvars.ContextVar[UsageRecord | None] = contextvars.ContextVar(
    "current_usage", default=None
)

# Process-wide totals, for the budget ceiling.
_TOTALS_LOCK = threading.Lock()
_TOTALS = UsageRecord()
_RATE_LIMIT_HITS = 0


def start_task_usage() -> UsageRecord:
    """Begin accounting for one task. Returns the record to read when it ends."""
    record = UsageRecord()
    _CURRENT_USAGE.set(record)
    return record


def global_usage() -> UsageRecord:
    with _TOTALS_LOCK:
        snapshot = UsageRecord(
            model=_TOTALS.model,
            calls=_TOTALS.calls,
            prompt_tokens=_TOTALS.prompt_tokens,
            output_tokens=_TOTALS.output_tokens,
            thought_tokens=_TOTALS.thought_tokens,
            cached_tokens=_TOTALS.cached_tokens,
            max_prompt_tokens=_TOTALS.max_prompt_tokens,
            errors=_TOTALS.errors,
        )
    return snapshot


def rate_limit_hits() -> int:
    with _TOTALS_LOCK:
        return _RATE_LIMIT_HITS


def note_rate_limit() -> None:
    global _RATE_LIMIT_HITS
    with _TOTALS_LOCK:
        _RATE_LIMIT_HITS += 1


def reset_global_usage() -> None:
    global _TOTALS, _RATE_LIMIT_HITS
    with _TOTALS_LOCK:
        _TOTALS = UsageRecord()
        _RATE_LIMIT_HITS = 0


@dataclass
class CallUsage:
    """Token counts for one call, in the field names UsageRecord.add reads.

    Gemini reports these directly on `usage_metadata`. Other providers report
    their own names, so their adapter translates into this shape and the counts
    then accumulate through exactly the same code. One ledger means the budget
    ceiling means the same thing on every arm.
    """

    prompt_token_count: int = 0
    candidates_token_count: int = 0
    thoughts_token_count: int = 0
    cached_content_token_count: int = 0


def record_call(
    model: str,
    prompt_tokens: int,
    output_tokens: int,
    thought_tokens: int = 0,
    cached_tokens: int = 0,
) -> None:
    """Record one API call against the current task and the process totals.

    Provider-neutral entry point, used by adapters for providers whose usage
    object is not Gemini's. The Gemini path calls UsageRecord.add directly with
    the response's own `usage_metadata`, which this reduces to.
    """
    usage = CallUsage(
        prompt_token_count=prompt_tokens,
        candidates_token_count=output_tokens,
        thoughts_token_count=thought_tokens,
        cached_content_token_count=cached_tokens,
    )
    record = _CURRENT_USAGE.get()
    if record is not None:
        record.add(model, usage)
    with _TOTALS_LOCK:
        _TOTALS.add(model, usage)


def note_error() -> None:
    """Count one failed call against the current task and the process totals."""
    record = _CURRENT_USAGE.get()
    if record is not None:
        record.errors += 1
    with _TOTALS_LOCK:
        _TOTALS.errors += 1


# --------------------------------------------------------------------------
# Client construction
# --------------------------------------------------------------------------


def build_client(api_key: str | None = None) -> genai.Client:
    """Build a Gemini Developer API client from an API key.

    Accepts GEMINI_API_KEY or GOOGLE_API_KEY, matching the google-genai
    convention, so an existing environment keeps working.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "No API key found. Set GEMINI_API_KEY in the environment or in .env at "
            "the repository root. Copy .env.example to .env and paste the key there."
        )
    # A request timeout is mandatory, not a nicety. google-genai sets none by
    # default, and AgentDojo's tenacity retry fires only on exceptions, never on
    # a silent hang, so one stalled connection blocks its worker forever. A real
    # sweep lost eight hours to exactly that: the process sat at 0 percent CPU
    # with 101 of 1081 tasks outstanding and never resumed.
    #
    # Ten minutes is generous for a single call even on a thinking model, and it
    # converts a hang into a retryable error.
    return genai.Client(
        api_key=key,
        http_options=genai_types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )


# --------------------------------------------------------------------------
# Patching
# --------------------------------------------------------------------------

_PATCHED = False
_THINKING_CONFIG: genai_types.ThinkingConfig | None = None


def _build_thinking_config(mode: str, budget: int) -> genai_types.ThinkingConfig | None:
    """Translate the harness thinking_mode into a ThinkingConfig, or None.

    Returning None means no thinking_config is sent at all, so each model applies
    its own default. That is the as-deployed condition.
    """
    if mode == "as_deployed":
        return None
    if mode == "normalized":
        return genai_types.ThinkingConfig(thinking_budget=budget)
    if mode == "off":
        return genai_types.ThinkingConfig(thinking_budget=0)
    raise ValueError(
        f"Unknown thinking_mode {mode!r}. Use as_deployed, normalized, or off."
    )


def install(thinking_mode: str = "as_deployed", thinking_budget: int = 4096) -> None:
    """Install the usage-accounting and thinking-configuration patch.

    Idempotent, so importing from several entry points is safe.
    """
    global _PATCHED, _THINKING_CONFIG
    _THINKING_CONFIG = _build_thinking_config(thinking_mode, thinking_budget)

    if _PATCHED:
        return

    original = google_llm.chat_completion_request

    def wrapped(model, client, contents, generation_config):
        if _THINKING_CONFIG is not None:
            # GenerateContentConfig is a pydantic model built fresh on every
            # call inside GoogleLLM.query, so mutating it here is safe.
            generation_config.thinking_config = _THINKING_CONFIG

        try:
            response = original(model, client, contents, generation_config)
        except Exception as exc:  # noqa: BLE001 - re-raised immediately
            if _is_rate_limit(exc):
                note_rate_limit()
            record = _CURRENT_USAGE.get()
            if record is not None:
                record.errors += 1
            with _TOTALS_LOCK:
                _TOTALS.errors += 1
            raise

        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            record = _CURRENT_USAGE.get()
            if record is not None:
                record.add(model, usage)
            with _TOTALS_LOCK:
                _TOTALS.add(model, usage)
        return response

    google_llm.chat_completion_request = wrapped
    _install_signature_relay()
    _PATCHED = True


# --------------------------------------------------------------------------
# Thought-signature relay, required by Gemini 3.x
# --------------------------------------------------------------------------

# Gemini 3 attaches an encrypted `thought_signature` to each function-call part
# and requires the client to echo it back on every subsequent turn. AgentDojo
# 0.1.35 predates the requirement: it rebuilds assistant turns from its own
# ChatAssistantMessage through `Part.from_function_call(name, args)`, which
# carries no signature. Every multi-turn tool-calling request to a 3.x model
# therefore fails with
#
#   400 INVALID_ARGUMENT: Function call is missing a thought_signature in
#   functionCall parts.
#
# Measured effect before the fix: 1081 of 1081 tasks failed on
# gemini-3.1-flash-lite while gemini-2.5-flash-lite completed cleanly, so the
# fault splits exactly along the 2.5/3.x boundary and would have taken out six
# of the nine models in the matrix.
#
# The relay stores each signature on the assistant message itself, so it
# travels with the conversation exactly as Google requires, and re-attaches it
# when the message is converted back. Signatures are bytes and AgentDojo writes
# every message to JSON, so they are base64-encoded in transit; storing the raw
# bytes would break the trace writer.

_SIGNATURE_KEY = "_thought_signatures"


def _install_signature_relay() -> None:
    original_to_assistant = google_llm._google_to_assistant_message
    original_parts_from = google_llm._parts_from_assistant_message

    def to_assistant(response):
        message = original_to_assistant(response)
        signatures: list[str | None] = []
        candidates = getattr(response, "candidates", None) or []
        if candidates and candidates[0].content and candidates[0].content.parts:
            for part in candidates[0].content.parts:
                if getattr(part, "function_call", None) is None:
                    continue
                raw = getattr(part, "thought_signature", None)
                signatures.append(base64.b64encode(raw).decode() if raw else None)
        if any(s is not None for s in signatures):
            message[_SIGNATURE_KEY] = signatures  # type: ignore[typeddict-unknown-key]
        return message

    def parts_from(assistant_message):
        parts = original_parts_from(assistant_message)
        signatures = assistant_message.get(_SIGNATURE_KEY) or []
        if not signatures:
            return parts
        # original_parts_from emits text parts first, then one part per tool
        # call in order, and to_assistant recorded signatures in that same
        # order, so positional pairing is exact.
        call_parts = [p for p in parts if getattr(p, "function_call", None) is not None]
        for part, signature in zip(call_parts, signatures):
            if signature:
                part.thought_signature = base64.b64decode(signature)
        return parts

    google_llm._google_to_assistant_message = to_assistant
    google_llm._parts_from_assistant_message = parts_from


def _is_rate_limit(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if code in (429, "429"):
        return True
    text = str(exc).lower()
    return "429" in text or "resource_exhausted" in text or "rate limit" in text


def register_model_names(model_ids: list[str]) -> None:
    """Teach AgentDojo's attack layer the prose name for each model.

    `get_model_name_from_pipeline` resolves the pipeline name against
    MODEL_NAMES by substring match and raises when nothing matches. Every entry
    maps to the same prose string that AgentDojo already uses for Gemini, so the
    injected attack text does not vary across the models under comparison.
    """
    for model_id in model_ids:
        MODEL_NAMES.setdefault(model_id, GOOGLE_MODEL_PROSE_NAME)


def build_llm(
    model_id: str,
    client: genai.Client,
    temperature: float = 0.0,
    max_output_tokens: int = 65535,
) -> google_llm.GoogleLLM:
    """Construct a GoogleLLM bound to an API-key client, bypassing ModelsEnum."""
    llm = google_llm.GoogleLLM(
        model=model_id,
        client=client,
        temperature=temperature,
        max_tokens=max_output_tokens,
    )
    llm.name = model_id
    return llm
