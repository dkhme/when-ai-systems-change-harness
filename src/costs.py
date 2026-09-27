# SPDX-License-Identifier: Apache-2.0
"""Pricing for the Google arm, computed from measured token counts.

Every figure comes from config/models_google.yaml, which carries the date it was
verified against Google's published pricing. Nothing here estimates: the inputs
are the `usage_metadata` counts returned by the API on each call.

Two properties of Gemini pricing are handled explicitly.

Long-context tiering. The Pro models charge roughly double above a 200,000-token
prompt. AgentDojo trajectories chain up to 18 tool calls with observations of
several thousand tokens each, and the conversation is resent in full on every
step, so the threshold is reachable on the Workspace suite. Pricing per call
against that call's own prompt size is the only correct treatment; pricing the
task total against a single tier would be wrong in both directions.

Implicit caching. Gemini discounts repeated prompt prefixes automatically and
reports the discounted portion as `cached_content_token_count`. We price cached
tokens at full rate, which makes every figure an upper bound. An upper bound is
the right error direction for a budget ceiling, and the cached counts are
recorded separately so the realised saving stays visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


@dataclass(frozen=True)
class ModelPricing:
    model_id: str
    lineage: str
    order: int
    input: float
    output: float
    input_long: float | None = None
    output_long: float | None = None
    long_context_threshold: int | None = None
    status: str = "stable"

    def _rates(self, prompt_tokens: int) -> tuple[float, float]:
        if (
            self.long_context_threshold is not None
            and self.input_long is not None
            and self.output_long is not None
            and prompt_tokens > self.long_context_threshold
        ):
            return self.input_long, self.output_long
        return self.input, self.output

    def cost_for_call(self, prompt_tokens: int, output_tokens: int) -> float:
        """Cost of one API call, tiered against that call's own prompt size."""
        in_rate, out_rate = self._rates(prompt_tokens)
        return (prompt_tokens * in_rate + output_tokens * out_rate) / 1_000_000


def load_registry(path: Path | None = None) -> dict[str, Any]:
    path = path or CONFIG_DIR / "models_google.yaml"
    with path.open() as f:
        return yaml.safe_load(f)


def load_pricing(path: Path | None = None) -> dict[str, ModelPricing]:
    registry = load_registry(path)
    pricing: dict[str, ModelPricing] = {}
    for model_id, spec in registry["models"].items():
        pricing[model_id] = ModelPricing(
            model_id=model_id,
            lineage=spec["lineage"],
            order=spec["order"],
            input=float(spec["input"]),
            output=float(spec["output"]),
            input_long=float(spec["input_long"]) if "input_long" in spec else None,
            output_long=float(spec["output_long"]) if "output_long" in spec else None,
            long_context_threshold=spec.get("long_context_threshold"),
            status=spec.get("status", "stable"),
        )
    return pricing


def cost_of_usage(pricing: ModelPricing, usage: Any) -> float:
    """Cost of one task from its UsageRecord.

    Output is charged on candidates plus thoughts, since thinking tokens bill at
    the output rate. Where per-call prompt sizes were recorded, each call is
    priced against its own tier and the output is apportioned across calls in
    proportion to prompt size, which is the closest attribution available
    without per-call output counts.
    """
    total_output = usage.output_tokens + usage.thought_tokens
    per_call = getattr(usage, "per_call_prompt_tokens", None)

    if not per_call:
        return pricing.cost_for_call(usage.prompt_tokens, total_output)

    prompt_total = sum(per_call) or 1
    cost = 0.0
    for prompt_tokens in per_call:
        share = prompt_tokens / prompt_total
        cost += pricing.cost_for_call(prompt_tokens, total_output * share)
    return cost


def crossed_long_context(pricing: ModelPricing, usage: Any) -> bool:
    """Whether any single call on this task priced at the long-context tier."""
    if pricing.long_context_threshold is None:
        return False
    return usage.max_prompt_tokens > pricing.long_context_threshold


def format_usd(amount: float) -> str:
    if amount < 0.01:
        return f"${amount:.4f}"
    if amount < 1:
        return f"${amount:.3f}"
    return f"${amount:,.2f}"
