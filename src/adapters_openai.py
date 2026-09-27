# SPDX-License-Identifier: Apache-2.0
"""Adapt AgentDojo 0.1.35 to the OpenAI Responses API.

Mirrors the function surface of `adapters.py`, `adapters_baseten.py` and
`adapters_anthropic.py` so `run_sweep.py --arm openai` binds this module and
changes nothing else.

Unlike the other three arms, this one cannot use the LLM class the package
ships. `OpenAILLM` speaks `/v1/chat/completions`, and the gpt-5.6 family cannot
use function tools on that endpoint at any reasoning effort other than 'none':

    Function tools with reasoning_effort are not supported for gpt-5.6-sol in
    /v1/chat/completions. To use function tools, use /v1/responses or set
    reasoning_effort to 'none'.

gpt-5.5 has no such restriction and reasons happily with tools on the old
endpoint. Running the arm on the package as it stands would therefore put
gpt-5.5 at its default reasoning effort and every 5.6 model at none, varying
reasoning and version together across the one within-tier version transition the
arm exists to measure. That is not a wire detail, it voids the primary cell.

So this module supplies `ResponsesLLM`, a pipeline element that speaks
`/v1/responses`, where all four models reason and call tools normally. Verified
against all four on 2026-08-17.

Four further things need handling.

**1. Reasoning tokens must not be double counted.** The Responses API reports
`usage.output_tokens` INCLUSIVE of reasoning tokens, whereas the shared ledger
follows Gemini's convention, where `candidates_token_count` and
`thoughts_token_count` are disjoint and `costs.py` charges their sum. Recording
both as reported would bill every reasoning token twice. `_record_usage`
subtracts before recording, and the subtraction is the single most consequential
line in this file.

**2. Temperature 0 is rejected by all four models.** Verified live 2026-08-17:

    Unsupported value: 'temperature' does not support 0 with this model.
    Only the default (1) value is supported.

Identical in consequence to what `claude-sonnet-5` forced on the Anthropic arm.
The arm runs at the default, the field is never sent, and `build_llm` refuses
any other temperature rather than letting a run proceed under a value the API
silently ignored. Determinism is unavailable, so the repetition floor is
mandatory and a single repetition reports one draw.

**3. Reasoning items are NOT carried across tool turns.** OpenAI recommends
echoing the returned `reasoning` items back on the following turn so the chain
survives a tool call. This adapter does not, for two reasons. It matches the
other three arms, none of which feed reasoning back either: the Gemini arm
reasons server-side per turn, and the Anthropic arm has thinking off entirely.
And carrying them needs per-call state on an object the runner shares across
worker threads, which is a race worth avoiding for a benchmark whose tasks
finish in two or three turns.

The cost is that these models may be measured slightly below their true
capability. That direction matters, because the capability gate is load-bearing
on this arm: a gate that looks empty could be this choice rather than the model.
If the gate comes back low, revisit this before concluding anything, and say so
in the write-up either way.

**4. `_REASONING_MODELS` in the package is the literal set {"o1", "o3"}**, so
nothing in `openai_llm.py` recognises a gpt-5.x identifier. Irrelevant here
because that module is bypassed, and recorded so nobody reintroduces it.

One thing deliberately NOT done: `store=True` with `previous_response_id`. It
would carry reasoning for free, but it persists every benchmark trajectory on
the vendor's servers and makes a run depend on state we do not hold. Requests go
out with `store=False`.

Prompt caching is not opt-in on this API. OpenAI caches prompts above 1024
tokens automatically and there is no flag to disable it, so unlike the other
three arms this one cannot choose to forgo it. Cached tokens are still priced at
the FULL input rate, which keeps every cost figure an upper bound and keeps the
convention identical across arms. The measured cache fraction is recorded in
each manifest so the gap between ceiling and invoice stays visible.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import openai
import yaml
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.attacks.base_attacks import DEFAULT_MODEL_NAME
from agentdojo.functions_runtime import EmptyEnv, Env, Function, FunctionCall, FunctionsRuntime
from agentdojo.models import MODEL_NAMES
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    MessageContentBlock,
    text_content_block_from_string,
)
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_random_exponential

import adapters
from adapters import note_error, note_rate_limit, record_call

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
REGISTRY_PATH = CONFIG_DIR / "models_openai.yaml"

REQUEST_TIMEOUT_S = 600.0
MAX_RETRIES = 3

# See point 2 in the module docstring. Every model in this arm rejects 0.0 and
# accepts only its default, so the field is never sent and this is the value the
# run must be recorded as having used.
DEFAULT_TEMPERATURE = 1.0

# Identical to the Anthropic and Baseten arms. Imported rather than re-spelled so
# the injected bytes cannot drift apart between arms. Registering "GPT-4", which
# is what the package ships for the gpt-4 family, would inject different bytes
# and silently break the cross-arm comparison.
OPENAI_MODEL_PROSE_NAME = DEFAULT_MODEL_NAME

_PATCHED = False
_PATCH_LOCK = threading.Lock()

# Bound by `install()` from the harness thinking mode. Module-level because
# `build_llm` takes the same arguments on every arm and cannot grow a parameter
# without changing all four. None means the model's own default, which is the
# `as_deployed` condition; 'none' is the `off` condition.
_REASONING_EFFORT: str | None = None


def load_registry(path: Path | None = None) -> dict[str, Any]:
    """Read the model registry. Never trusted on its own; see the file header."""
    with open(path or REGISTRY_PATH) as fh:
        return yaml.safe_load(fh)


def base_url() -> str:
    """Reported in the manifest so a run records the endpoint it actually hit."""
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")


def pipeline_name(model_id: str) -> str:
    """OpenAI ids carry no path separator, so the id is its own alias."""
    return model_id


def build_client(api_key: str | None = None) -> openai.OpenAI:
    """Build the client the pipeline holds.

    Unlike the Anthropic arm there is no per-call rebuild here. That arm needs
    one because `AnthropicLLM.query` runs `asyncio.run` per turn and binds an
    async connection pool to a dead event loop. This element is synchronous, so
    one thread-safe client is shared and the failure mode does not arise.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in the environment or in .env at "
            "the repository root, which is gitignored. Never paste a key into a "
            "source file or a manifest."
        )
    kwargs: dict[str, Any] = {
        "api_key": key,
        "timeout": REQUEST_TIMEOUT_S,
        "max_retries": MAX_RETRIES,
    }
    if "OPENAI_BASE_URL" in os.environ:
        kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
    return openai.OpenAI(**kwargs)


def _is_rate_limit(exc: Exception) -> bool:
    if isinstance(exc, openai.RateLimitError):
        return True
    return getattr(exc, "status_code", None) == 429


def install(thinking_mode: str = "as_deployed", thinking_budget: int = 4096) -> None:
    """Validate the reasoning condition. Idempotent and lock-guarded.

    Nothing is patched: this arm owns its pipeline element rather than wrapping
    one the package ships. The function exists because the runner calls it on
    every arm, and it is the right place to refuse a condition this provider
    cannot deliver.

    `normalized` is refused. The Responses API exposes reasoning effort as an
    ordinal level (none, low, medium, high) rather than a token budget, so there
    is no value that would make this arm's reasoning commensurate with the
    Gemini arm's `thinking_budget`. Pretending otherwise would put a number in a
    manifest that means something different from the same number elsewhere.

    `off` is permitted and maps to reasoning_effort='none'. It is a legitimate
    second condition, and it is also the only reasoning setting under which Chat
    Completions accepts function tools on the gpt-5.6 family (see the module
    docstring), which makes it the as-deployed condition for a practitioner on
    that endpoint.
    """
    global _PATCHED, _REASONING_EFFORT
    if thinking_mode == "normalized":
        raise RuntimeError(
            "thinking_mode='normalized' is not available on the OpenAI arm. This "
            "API takes an ordinal reasoning effort, not a token budget, so no "
            "setting here is commensurate with the Gemini arm's thinking_budget "
            "of 4096. Run as_deployed, or run off and label it as the off arm."
        )
    if thinking_mode not in ("as_deployed", "off"):
        raise RuntimeError(f"Unknown thinking_mode {thinking_mode!r}.")
    with _PATCH_LOCK:
        _REASONING_EFFORT = None if thinking_mode == "as_deployed" else "none"
        _PATCHED = True


def register_model_names(model_ids: list[str]) -> None:
    """Teach AgentDojo's attack layer a prose name for each model.

    `important_instructions` interpolates this string into the injected message
    and `get_model_name_from_pipeline` raises when nothing matches. Registering
    the name-neutral string keeps the injected bytes identical to the Anthropic
    and Baseten arms.
    """
    for model_id in model_ids:
        MODEL_NAMES.setdefault(model_id, OPENAI_MODEL_PROSE_NAME)
        MODEL_NAMES.setdefault(pipeline_name(model_id), OPENAI_MODEL_PROSE_NAME)


# --------------------------------------------------------------------------
# Message translation, AgentDojo <-> Responses API
# --------------------------------------------------------------------------


def _blocks_to_text(content: Sequence[MessageContentBlock] | None) -> str:
    """Flatten AgentDojo's content blocks to the plain string this API takes."""
    if not content:
        return ""
    return "".join(block.get("content") or "" for block in content)


def _function_to_responses_tool(f: Function) -> dict[str, Any]:
    """Responses function tools are FLAT, unlike Chat Completions' nested shape.

    `strict` is False because AgentDojo's tool schemas carry optional fields and
    defaults, which strict mode rejects. Strict mode would buy schema-guaranteed
    arguments at the cost of refusing most of the benchmark's own tools.
    """
    return {
        "type": "function",
        "name": f.name,
        "description": f.description,
        "parameters": f.parameters.model_json_schema(),
        "strict": False,
    }


def _messages_to_responses_input(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """Translate the conversation into Responses input items.

    An assistant turn that carried tool calls becomes one `function_call` item
    per call, preceded by a text item when the model also spoke. A tool result
    becomes a `function_call_output` keyed by the same `call_id`, which is the
    id AgentDojo stored on the FunctionCall when the turn was parsed.
    """
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "system":
            # Matches what the package's Chat Completions path does: the system
            # message goes in as `developer`, not `system`.
            items.append({"role": "developer", "content": _blocks_to_text(message["content"])})
        elif role == "user":
            items.append({"role": "user", "content": _blocks_to_text(message["content"])})
        elif role == "assistant":
            text = _blocks_to_text(message["content"])
            if text:
                items.append({"role": "assistant", "content": text})
            for tool_call in message["tool_calls"] or []:
                if tool_call.id is None:
                    raise ValueError("`tool_call.id` is required for the Responses API")
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call.id,
                        "name": tool_call.function,
                        "arguments": json.dumps(tool_call.args),
                    }
                )
        elif role == "tool":
            call_id = message["tool_call_id"]
            if call_id is None:
                raise ValueError("`tool_call_id` should be specified for the Responses API.")
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": message["error"] or _blocks_to_text(message["content"]),
                }
            )
        else:
            raise ValueError(f"Invalid message type: {message}")
    return items


def _response_to_assistant_message(response: Any) -> ChatAssistantMessage:
    """Read the assistant turn back out of a Responses result.

    `reasoning` items are read for their token count only and deliberately not
    carried forward; see point 3 in the module docstring.
    """
    tool_calls: list[FunctionCall] = []
    texts: list[str] = []
    for item in response.output or []:
        kind = getattr(item, "type", None)
        if kind == "function_call":
            tool_calls.append(
                FunctionCall(
                    function=item.name,
                    args=json.loads(item.arguments) if item.arguments else {},
                    id=item.call_id,
                )
            )
        elif kind == "message":
            for block in getattr(item, "content", None) or []:
                text = getattr(block, "text", None)
                if text:
                    texts.append(text)

    joined = "".join(texts)
    content = [text_content_block_from_string(joined)] if joined else None
    return ChatAssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls or None,
    )


def _record_usage(model: str, response: Any) -> None:
    """Translate Responses usage into the shared ledger.

    THE SUBTRACTION MATTERS. `usage.output_tokens` on this API INCLUDES
    reasoning tokens, while the ledger follows Gemini's convention where output
    and thought counts are disjoint and `costs.py` charges their sum. Recording
    both as reported would bill every reasoning token twice, and on this arm's
    flagships that is 30 dollars per million tokens of pure double count.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return

    out_details = getattr(usage, "output_tokens_details", None)
    in_details = getattr(usage, "input_tokens_details", None)
    reasoning = int(getattr(out_details, "reasoning_tokens", 0) or 0)
    cached = int(getattr(in_details, "cached_tokens", 0) or 0)
    total_output = int(getattr(usage, "output_tokens", 0) or 0)

    record_call(
        model=model,
        prompt_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=max(total_output - reasoning, 0),
        thought_tokens=reasoning,
        # Priced at the full input rate by `costs.py`, exactly as on the other
        # arms. Recorded so the gap between the ceiling and the invoice is
        # visible rather than inferred.
        cached_tokens=cached,
    )


@retry(
    wait=wait_random_exponential(multiplier=1, max=40),
    stop=stop_after_attempt(3),
    reraise=True,
    retry=retry_if_not_exception_type((openai.BadRequestError, openai.UnprocessableEntityError)),
)
def _responses_request(
    client: openai.OpenAI,
    model: str,
    items: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    reasoning_effort: str | None,
) -> Any:
    """One call, with the package's own retry policy.

    A BadRequestError is never retried. On this arm the two that occur are a
    rejected temperature and a rejected reasoning effort, and both are
    configuration errors that retrying converts into three identical failures
    and a slower run.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "input": items,
        # See the module docstring: no trajectory is persisted on the vendor's
        # servers, and nothing in a run depends on state we do not hold.
        "store": False,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if reasoning_effort is not None:
        kwargs["reasoning"] = {"effort": reasoning_effort}
    return client.responses.create(**kwargs)


class ResponsesLLM(BasePipelineElement):
    """AgentDojo pipeline element speaking `/v1/responses`.

    Args:
        client: the OpenAI client.
        model: the model id, for example `gpt-5.6-sol`.
        reasoning_effort: None leaves the model at its own default, which is the
            `as_deployed` condition. 'none' is the `off` condition.
    """

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        reasoning_effort: str | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.name = pipeline_name(model)

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        items = _messages_to_responses_input(messages)
        tools = [_function_to_responses_tool(tool) for tool in runtime.functions.values()]
        try:
            response = _responses_request(
                self.client, self.model, items, tools, self.reasoning_effort
            )
        except Exception as exc:  # noqa: BLE001 - counted, then re-raised
            if _is_rate_limit(exc):
                note_rate_limit()
            note_error()
            raise
        _record_usage(self.model, response)
        output = _response_to_assistant_message(response)
        return query, runtime, env, [*messages, output], extra_args


def build_llm(
    model_id: str,
    client: openai.OpenAI,
    temperature: float = 0.0,
    max_output_tokens: int | None = None,
) -> ResponsesLLM:
    """Construct a ResponsesLLM bound to this client.

    `max_output_tokens` is accepted for surface compatibility with the other
    arms and deliberately not sent. The harness config carries 65535 for Gemini;
    these models cap at 128000 and never approach it on an AgentDojo turn, which
    is a tool call plus a short preamble. Sending a cap we do not need is how the
    Anthropic arm nearly manufactured a result: a truncated turn emits no valid
    tool call, fails the user task, and, because a model that emits nothing also
    injects nothing, SCORES AS SECURE.
    """
    if temperature != DEFAULT_TEMPERATURE:
        raise RuntimeError(
            f"{model_id} rejects temperature={temperature} "
            f"('temperature does not support 0 with this model. Only the default "
            f"(1) value is supported'). All four models in this arm reject 0.0, so "
            f"the whole arm can only be compared at {DEFAULT_TEMPERATURE}. Run with "
            f"--temperature {DEFAULT_TEMPERATURE} --allow-config-change, and treat "
            f"the repetition floor as mandatory: temperature 1.0 is not "
            f"deterministic, so one repetition measures one draw."
        )
    if not _PATCHED:
        raise RuntimeError(
            "install() has not run, so the reasoning condition is unbound and "
            "this model would silently take the module default rather than the "
            "configured thinking_mode. The runner calls install() before "
            "build_llm(); a caller that skips it must do the same."
        )
    return ResponsesLLM(client=client, model=model_id, reasoning_effort=_REASONING_EFFORT)


# The runner reads these through the module surface; re-exported so every arm
# exposes the identical set.
start_task_usage = adapters.start_task_usage
global_usage = adapters.global_usage
rate_limit_hits = adapters.rate_limit_hits
reset_global_usage = adapters.reset_global_usage
