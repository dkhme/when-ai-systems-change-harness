# SPDX-License-Identifier: Apache-2.0
"""Freeze the harness configuration and enforce it across arms.

One invariant governs the whole study: the harness configuration must be frozen
before the first run and reused byte-identically for every arm, or the
cross-family comparison is void.
Deadline pressure is exactly what breaks such an invariant, so this module makes
it mechanical rather than a matter of discipline.

The hash covers everything that can change what the model is asked:

  - the harness section of config/harness.yaml, canonicalised
  - the generation section, including the thinking mode
  - the installed agentdojo version
  - the resolved system message text, not merely its name, so a change in the
    package's default message is caught rather than silently absorbed
  - the attack class and its jailbreak template
  - the task counts of the selected benchmark version

The hash is written to config/harness.lock.json and copied into every results
manifest. run_sweep.py compares the two before starting.
"""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
HARNESS_PATH = CONFIG_DIR / "harness.yaml"
LOCK_PATH = CONFIG_DIR / "harness.lock.json"


def load_harness_config(path: Path | None = None) -> dict[str, Any]:
    with (path or HARNESS_PATH).open() as f:
        return yaml.safe_load(f)


def _agentdojo_version() -> str:
    try:
        return version("agentdojo")
    except PackageNotFoundError:
        return "not-installed"


def _resolved_system_message(config: dict[str, Any]) -> str:
    from agentdojo.agent_pipeline.agent_pipeline import load_system_message

    harness = config["harness"]
    if harness.get("system_message"):
        return str(harness["system_message"])
    return load_system_message(harness.get("system_message_name"))


def _attack_fingerprint(config: dict[str, Any]) -> dict[str, Any]:
    from agentdojo.attacks.attack_registry import ATTACKS

    attack_name = config["harness"]["attack"]
    if attack_name is None:
        return {"name": None, "template": None}
    if attack_name not in ATTACKS:
        raise ValueError(f"Unknown attack {attack_name!r}. Known: {sorted(ATTACKS)}")
    attack_cls = ATTACKS[attack_name]
    template = getattr(attack_cls, "_JB_STRING", None)
    return {"name": attack_name, "template": template}


def _suite_fingerprint(config: dict[str, Any]) -> dict[str, Any]:
    from agentdojo.task_suite.load_suites import get_suite

    harness = config["harness"]
    fingerprint = {}
    for suite_name in harness["suites"]:
        suite = get_suite(harness["benchmark_version"], suite_name)
        fingerprint[suite_name] = {
            "user_tasks": sorted(suite.user_tasks.keys()),
            "injection_tasks": sorted(suite.injection_tasks.keys()),
        }
    return fingerprint


def compute_fingerprint(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_harness_config()
    return {
        "agentdojo_version": _agentdojo_version(),
        "harness": config["harness"],
        "generation": config["generation"],
        "resolved_system_message": _resolved_system_message(config),
        "attack": _attack_fingerprint(config),
        "suites": _suite_fingerprint(config),
    }


def compute_hash(fingerprint: dict[str, Any] | None = None) -> str:
    fingerprint = fingerprint if fingerprint is not None else compute_fingerprint()
    canonical = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def write_lock(path: Path | None = None) -> dict[str, Any]:
    fingerprint = compute_fingerprint()
    lock = {
        "config_hash": compute_hash(fingerprint),
        "fingerprint": fingerprint,
    }
    (path or LOCK_PATH).write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    return lock


def read_lock(path: Path | None = None) -> dict[str, Any] | None:
    path = path or LOCK_PATH
    if not path.exists():
        return None
    return json.loads(path.read_text())


def check(
    path: Path | None = None, config: dict[str, Any] | None = None
) -> tuple[bool, str, str | None]:
    """Compare the current configuration against the lock.

    Returns (matches, current_hash, locked_hash). A missing lock reports
    matches=False with locked_hash None, which run_sweep.py treats as
    "not yet frozen" rather than as a violation.

    `config` must be the EFFECTIVE configuration, after any command-line
    override has been applied. Hashing the file instead would let an override
    such as --attack or --defense change what is sent to the model while the
    hash continued to certify the unmodified file, which is the one failure
    this whole mechanism exists to prevent.
    """
    current = compute_hash(compute_fingerprint(config) if config is not None else None)
    lock = read_lock(path)
    if lock is None:
        return False, current, None
    return lock["config_hash"] == current, current, lock["config_hash"]


def _diff_report(lock: dict[str, Any]) -> list[str]:
    """Name the fields that moved since the lock was written."""
    current = compute_fingerprint()
    locked = lock["fingerprint"]
    lines: list[str] = []

    def walk(a: Any, b: Any, path: str) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                walk(a.get(key), b.get(key), f"{path}.{key}" if path else key)
        elif a != b:
            shown_a = str(a)
            shown_b = str(b)
            if len(shown_a) > 60:
                shown_a = shown_a[:57] + "..."
            if len(shown_b) > 60:
                shown_b = shown_b[:57] + "..."
            lines.append(f"  {path}: locked={shown_b!r} -> current={shown_a!r}")

    walk(current, locked, "")
    return lines


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Freeze or verify the harness configuration.")
    parser.add_argument(
        "action",
        nargs="?",
        default="check",
        choices=["check", "write"],
        help="check compares against the lock; write creates or replaces it.",
    )
    args = parser.parse_args()

    if args.action == "write":
        lock = write_lock()
        print(f"Wrote {LOCK_PATH}")
        print(f"config_hash = {lock['config_hash']}")
        print()
        print("Every arm of the study must now run against this hash.")
        return 0

    matches, current, locked = check()
    if locked is None:
        print(f"No lock file at {LOCK_PATH}.")
        print(f"Current config_hash would be {current}.")
        print("Run `python src/freeze.py write` to freeze it.")
        return 1
    if matches:
        print(f"Configuration matches the lock. config_hash = {current}")
        return 0

    print("CONFIGURATION HAS CHANGED SINCE IT WAS FROZEN.")
    print(f"  locked  : {locked}")
    print(f"  current : {current}")
    lock = read_lock()
    assert lock is not None
    for line in _diff_report(lock):
        print(line)
    print()
    print("Results recorded under the locked hash are not comparable with results")
    print("produced under this configuration. Either revert the change, or re-freeze")
    print("and re-run every arm.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
