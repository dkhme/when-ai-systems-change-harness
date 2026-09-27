# SPDX-License-Identifier: Apache-2.0
"""Compare the open-weight arm against the Gemini contamination probe.

Both arms ran `important_instructions_no_names` over Slack, Travel and Banking at
two repetitions under configuration `28f36d30e3a053c1`, so the injected payload
is byte-identical and the 389 cases are the same 389 cases. That is what makes a
cross-family comparison legitimate here and not elsewhere in the study.

Three tables are produced.

**Attack success rate**, POOLED per trace rather than collapsed over repetitions.
The distinction matters: collapsing runs slightly higher on split-prone models,
because a one-to-one tie across two repetitions resolves to insecure. Quoting a
collapsed rate against a pooled table would manufacture a difference out of the
convention.

**Paired version transitions**, which are the contribution. Collapsed, because a
paired metric needs one outcome per case. Reported three ways: raw, restricted
to cases both versions solved, and restricted to injection goals both versions
can demonstrably execute when asked directly. The third is the one to quote.

**The noise floor**, measured per model by pairing each model against ITSELF
across its two repetitions. This floor need not be small, so it is measured
rather than assumed. A regression that does not clear its own model's floor is
not a regression, and no amount of significance testing substitutes for having
measured it.

Usage:
    python src/compare_arms.py results/baseten_<run_id>
    python src/compare_arms.py results/baseten_<run_id> --json-out cross_arm.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import sys
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import analyze  # noqa: E402

EXPORT_ROOT = SRC.parent / "results_export"
GEMINI_PROBE = EXPORT_ROOT / "probe_no_names" / "per_case.csv.gz"

CaseKey = tuple[str, str, str]


class Arm:
    """One arm's per-case outcomes, keyed by model."""

    def __init__(self, name: str) -> None:
        self.name = name
        # model -> case -> list of per-repetition secure votes
        self.votes: dict[str, dict[CaseKey, list[bool]]] = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
        # model -> case -> list of per-repetition utility votes
        self.utility: dict[str, dict[CaseKey, list[bool]]] = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
        # model -> (suite, injection_task) -> list of votes on "goal executable"
        self.goals: dict[str, dict[tuple[str, str], list[bool]]] = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
        self.excluded: dict[str, int] = collections.Counter()

    def add_case(self, model: str, key: CaseKey, secure: bool, utility: bool) -> None:
        self.votes[model][key].append(secure)
        self.utility[model][key].append(utility)

    def add_goal(self, model: str, suite: str, injection_task: str, achieved: bool) -> None:
        self.goals[model][(suite, injection_task)].append(achieved)

    @property
    def models(self) -> list[str]:
        return sorted(self.votes)

    def pooled_asr(self, model: str) -> tuple[float, int, int]:
        """Insecure traces over all traces. The comparable figure."""
        traces = [v for votes in self.votes[model].values() for v in votes]
        insecure = sum(1 for v in traces if not v)
        return (insecure / len(traces) if traces else 0.0), insecure, len(traces)

    def collapsed(self, model: str) -> dict[CaseKey, bool]:
        return {k: (sum(v) * 2 > len(v)) for k, v in self.votes[model].items()}

    def collapsed_utility(self, model: str) -> dict[CaseKey, bool]:
        return {k: (sum(v) * 2 > len(v)) for k, v in self.utility[model].items()}

    def executable_goals(self, model: str) -> set[tuple[str, str]]:
        return {k for k, v in self.goals[model].items() if sum(v) * 2 > len(v)}

    def split_count(self, model: str) -> int:
        return sum(1 for v in self.votes[model].values() if len(set(v)) > 1)

    def noise_floor(self, model: str) -> tuple[int, int]:
        """Spurious negative flips from pairing the model against itself.

        Only defined with at least two repetitions. Returns (flips, cases).
        """
        pairs = [(k, v) for k, v in self.votes[model].items() if len(v) >= 2]
        flips = sum(1 for _, v in pairs if v[0] and not v[1])
        return flips, len(pairs)


def load_gemini_probe() -> Arm:
    arm = Arm("gemini")
    if not GEMINI_PROBE.exists():
        raise FileNotFoundError(f"Gemini probe export not found at {GEMINI_PROBE}")
    with gzip.open(GEMINI_PROBE, "rt") as fh:
        for row in csv.DictReader(fh):
            model = row["model"]
            suite, user_task = row["suite"], row["user_task"]
            injection = row["injection_task"]
            utility = bool(int(row["utility"]))
            if not injection:
                # The probe export does not distinguish an injection task run as
                # a standalone user task from an ordinary user task, so the
                # attack-capability gate is recovered from the id convention,
                # which is how AgentDojo names them.
                if user_task.startswith("injection_task"):
                    arm.add_goal(model, suite, user_task, utility)
                continue
            arm.add_case(model, (suite, user_task, injection), bool(int(row["secure"])), utility)
    return arm


def load_run(run_dir: Path) -> Arm:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    attack = (manifest.get("harness") or {}).get("attack")
    aliases: dict[str, str] = manifest.get("model_aliases") or {}
    models = [m for m in (sorted(aliases.values()) or []) if (run_dir / m).is_dir()]
    if not models:
        models = sorted(d.name for d in run_dir.iterdir() if d.is_dir() and any(d.glob("rep*")))

    arm = Arm(manifest.get("arm", run_dir.name))
    for model in models:
        for rep_dir in sorted(d for d in (run_dir / model).iterdir() if d.name.startswith("rep")):
            for path in rep_dir.rglob("*.json"):
                if path.name == "manifest.json":
                    continue
                try:
                    data = json.loads(path.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                if not isinstance(data, dict) or "suite_name" not in data:
                    continue
                injection = data.get("injection_task_id")
                # A crashed task leaves a trace with no outcome keys. Excluded
                # and counted, never scored.
                if not isinstance(data.get("utility"), bool):
                    arm.excluded[model] += 1
                    continue
                if injection is None:
                    if str(data["user_task_id"]).startswith("injection_task"):
                        arm.add_goal(model, data["suite_name"], data["user_task_id"], data["utility"])
                    continue
                if not isinstance(data.get("security"), bool):
                    arm.excluded[model] += 1
                    continue
                if attack and data.get("attack_type") != attack:
                    continue
                arm.add_case(
                    model,
                    (data["suite_name"], data["user_task_id"], injection),
                    # AgentDojo's `security` is True when the ATTACK SUCCEEDED.
                    not data["security"],
                    data["utility"],
                )
    return arm


def transition(arm: Arm, before: str, after: str) -> dict[str, Any]:
    """Paired statistics for one version transition, gated three ways."""
    sec_b, sec_a = arm.collapsed(before), arm.collapsed(after)
    util_b, util_a = arm.collapsed_utility(before), arm.collapsed_utility(after)
    shared = set(sec_b) & set(sec_a)

    gates = {
        "raw": shared,
        "utility_matched": {k for k in shared if util_b.get(k) and util_a.get(k)},
        "attack_capability_matched": {
            k for k in shared if (k[0], k[2]) in (arm.executable_goals(before) & arm.executable_goals(after))
        },
    }

    out: dict[str, Any] = {"before": before, "after": after, "gates": {}}
    for gate, keys in gates.items():
        neg = sum(1 for k in keys if sec_b[k] and not sec_a[k])
        pos = sum(1 for k in keys if not sec_b[k] and sec_a[k])
        secure_before = sum(1 for k in keys if sec_b[k])
        p, _ = analyze._mcnemar(neg, pos)
        out["gates"][gate] = {
            "n": len(keys),
            "asr_before": round(1 - (secure_before / len(keys)), 4) if keys else 0.0,
            "asr_after": round(1 - (sum(1 for k in keys if sec_a[k]) / len(keys)), 4) if keys else 0.0,
            "negative_flips": neg,
            "positive_flips": pos,
            "secure_before": secure_before,
            "nfr": round(neg / secure_before, 4) if secure_before else 0.0,
            "mcnemar_p": p,
        }
    floor, floor_n = arm.noise_floor(before)
    out["noise_floor_of_before_model"] = {"spurious_negative_flips": floor, "cases": floor_n}
    raw_neg = out["gates"]["raw"]["negative_flips"]
    out["clears_noise_floor"] = bool(floor_n) and raw_neg > floor
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", type=Path, help="A results/baseten_<run_id> directory.")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else Path.cwd() / args.run_dir
    if not run_dir.exists():
        print(f"No such run directory: {run_dir}")
        return 1

    open_weight = load_run(run_dir)
    gemini = load_gemini_probe()

    print("=" * 84)
    print("CROSS-ARM COMPARISON: open weights against the Gemini contamination probe")
    print("=" * 84)
    print("Both arms: important_instructions_no_names, slack+travel+banking, 2 reps,")
    print("temperature 0, no defense, config 28f36d30e3a053c1. Identical injected bytes.")

    print("\nAttack success rate, POOLED per trace (not collapsed over repetitions)")
    print(f"  {'model':<30}{'arm':<8}{'ASR':>8}{'95% CI':>18}{'traces':>9}{'split':>7}{'excl':>6}")
    print("  " + "-" * 86)
    for arm in (open_weight, gemini):
        for model in arm.models:
            asr, insecure, traces = arm.pooled_asr(model)
            low, high = analyze._confint(insecure, traces)
            label = "open" if arm is open_weight else "gemini"
            print(
                f"  {model:<30}{label:<8}{asr:>7.1%}{f'[{low:.1%}, {high:.1%}]':>18}"
                f"{traces:>9}{arm.split_count(model):>7}{arm.excluded.get(model, 0):>6}"
            )
        print()

    print("Noise floor: each model paired against ITSELF across its two repetitions.")
    print("A transition whose negative flips do not exceed this is not distinguishable")
    print("from resampling the same model twice.")
    print(f"  {'model':<30}{'spurious neg flips':>20}{'cases':>8}")
    print("  " + "-" * 60)
    for arm in (open_weight, gemini):
        for model in arm.models:
            flips, n = arm.noise_floor(model)
            if n:
                print(f"  {model:<30}{flips:>20}{n:>8}")

    # Transitions within each arm, ordered by the registry's lineage order.
    results: dict[str, Any] = {"transitions": []}
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    import costs

    pricing = costs.load_pricing(
        SRC.parent / "config" / manifest.get("model_registry", "models_baseten.yaml")
    )
    aliases = manifest.get("model_aliases") or {}
    alias_to_id = {a: s for s, a in aliases.items()}
    lineages: dict[str, list[str]] = collections.defaultdict(list)
    for model in open_weight.models:
        served = alias_to_id.get(model, model)
        if served in pricing:
            lineages[pricing[served].lineage].append(model)
    for chain in lineages.values():
        chain.sort(key=lambda m: pricing[alias_to_id.get(m, m)].order)

    print("\n\nPaired version transitions, open-weight arm")
    print("Collapsed over repetitions. The attack-capability-matched row is the one to quote.")
    for lineage, chain in sorted(lineages.items()):
        for before, after in zip(chain, chain[1:]):
            t = transition(open_weight, before, after)
            results["transitions"].append(t)
            floor = t["noise_floor_of_before_model"]["spurious_negative_flips"]
            print(f"\n  {lineage}:  {before}  ->  {after}")
            print(f"    {'gate':<30}{'n':>6}{'ASR':>17}{'neg':>6}{'pos':>6}{'NFR':>8}{'p':>11}")
            print("    " + "-" * 84)
            for gate, g in t["gates"].items():
                p = "n/a" if g["mcnemar_p"] is None else f"{g['mcnemar_p']:.2g}"
                shift = f"{g['asr_before']:.1%} -> {g['asr_after']:.1%}"
                print(
                    f"    {gate:<30}{g['n']:>6}{shift:>17}"
                    f"{g['negative_flips']:>6}{g['positive_flips']:>6}{g['nfr']:>7.1%}{p:>11}"
                )
            verdict = (
                "REGRESSION clears the noise floor"
                if t["clears_noise_floor"]
                else f"negative flips do NOT clear this model's own floor of {floor}"
            )
            print(f"    verdict: {verdict}")

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
