# When AI Systems Change harness

This repository contains the experiment harness that accompanies the paper
*When AI Systems Change: A Survey of Security Reassessment across Inheritance,
Updates, and Runtime Binding*. The harness runs the AgentDojo prompt-injection
benchmark against successive versions of hosted language models and compares
the versions case by case, so that a case which was secure under one version
and insecure under the next is counted rather than averaged away.

The release contains code and configuration only. It contains no API keys, no
raw traces and no per-case results; see [What this release does not
contain](#what-this-release-does-not-contain).

## What the harness is

**Benchmark.** The harness uses AgentDojo 0.1.35 as a pinned, unmodified
dependency and runs benchmark version v1.2.2. Over its four suites (workspace,
slack, travel, banking) that version has 97 user tasks and 35 injection tasks,
so one pass is 949 security cases (user task by injection task) plus 132
utility runs, or 1081 task runs per model and repetition. Restricted to slack,
travel and banking it is 389 security cases and 78 utility runs.

**Runner.** `src/run_sweep.py` schedules each (user task, injection task) pair
as its own work unit, writes every repetition to its own directory, reads the
token usage the provider reports on each call, prices it against the model
registry, and stops at a hard budget ceiling. A run resumes from AgentDojo's
per-task cache when it is started again with the same run identifier.

**Frozen configuration.** `src/freeze.py` hashes everything that can change
what a model is asked: the `harness` and `generation` sections of
`config/harness.yaml`, the installed AgentDojo version, the resolved text of
the system message, the attack name and its injection template, and the task
identifiers of every configured suite. The hash (the first 16 hexadecimal
digits of a SHA-256 over canonical JSON) is stored in
`config/harness.lock.json`. `run_sweep.py` hashes the effective configuration
after command-line overrides and refuses to start when it differs from the
lock, unless `--allow-config-change` is given; every run manifest records the
hash and whether it matched.

**Provider adapters.** Each arm binds one adapter module with the same
function surface, so the runner itself does not branch by provider.

| Module | Arm | What it changes relative to the released AgentDojo package |
|---|---|---|
| `src/adapters.py` | `google` (Gemini Developer API) | Builds an API-key client instead of the Vertex AI client, sets a request timeout, records token usage, applies the thinking configuration, and relays the `thought_signature` that Gemini 3.x requires on function-call turns. |
| `src/adapters_baseten.py` | `baseten` (open-weight models behind Baseten's OpenAI-compatible endpoint) | Sends the system message as `role=system` and flattens content parts into a string, corrects AgentDojo 0.1.35's dropping of a temperature of exactly 0, records token usage, and maps served identifiers containing `/` to one path segment. |
| `src/adapters_anthropic.py` | `anthropic` (Messages API) | Raises `max_tokens` from the package default of 1024 to 8192, builds a fresh asynchronous client per call, records token usage, and permits only the `as_deployed` thinking mode, which on this API means extended thinking is off. |
| `src/adapters_openai.py` | `openai` (Responses API) | Supplies its own pipeline element for `/v1/responses`, records reasoning tokens separately from output tokens, sends requests with `store=False`, and refuses any temperature other than 1.0. |

**Analysis.** The analysis scripts compute attack success rates with Wilson
intervals, the security negative-flip rate between versions (negative flips
divided by the cases secure under the earlier version), McNemar tests on the
discordant pairs, per-model noise floors from repeated runs of the same model,
and capability-gated variants of each comparison.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Adapters, runner, cost model, configuration freeze, self-tests, live pre-flight checks and analysis scripts. |
| `config/harness.yaml` | The frozen harness configuration. |
| `config/harness.lock.json` | The configuration hash `24a5898747a51439` and the fingerprint it was computed from. |
| `config/models_*.yaml` | Model registries for the four arms: identifiers, lineage order, prices, presets and the date each was verified. |
| `runners/` | Two multi-run shell sequences (zsh) and their README. |
| `requirements.txt`, `setup.sh`, `.env.example` | Installation and credential template. |
| `.gitignore` | Keeps `.env` files, the virtual environment, bytecode, run outputs and logs out of version control. |
| `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md` | Licensing and attribution. |
| `MANIFEST.sha256` | SHA-256 of every other file in the release. |

## Installation

The offline self-tests below pass with Python 3.12.5 and agentdojo 0.1.35,
google-genai 2.17.0, openai 2.53.0, anthropic 0.120.2, tenacity 9.1.4,
python-dotenv 1.2.2, PyYAML 6.0.3, statsmodels 0.14.6 and scipy 1.18.0. We
recommend Python 3.12; `setup.sh` itself accepts Python 3.10 or newer.

```sh
./setup.sh                    # or: PYTHON=python3.12 ./setup.sh
source .venv/bin/activate
```

`setup.sh` creates `.venv` in the repository root, installs
`requirements.txt`, and copies `.env.example` to `.env` if no `.env` exists.
The manual equivalent is:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Only `agentdojo` is pinned exactly. The OpenAI and Anthropic SDKs and tenacity
are installed as dependencies of AgentDojo rather than listed directly, and the
remaining requirements are lower bounds. Because the installed AgentDojo
version is part of the configuration hash, any other AgentDojo version changes
the hash, and `run_sweep.py` then refuses to run without
`--allow-config-change`.

## Credentials

The adapters read API keys from environment variables. `run_sweep.py` and the
`verify_setup*.py` scripts also load `.env` from the repository root through
python-dotenv; a variable already set in the environment takes precedence over
the file. The self-tests and the analysis scripts need no key.

| Arm | Required | Optional |
|---|---|---|
| `google` | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | none |
| `baseten` | `BASETEN_API_KEY` | `BASETEN_BASE_URL` |
| `anthropic` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |
| `openai` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |

`.env.example` contains empty placeholders only: `GEMINI_API_KEY=`, and
commented-out `ANTHROPIC_API_KEY=`, `OPENAI_API_KEY=` and `BASETEN_API_KEY=`
lines. Uncomment the line for each arm you run.

Never commit `.env` or any key. The included `.gitignore` excludes `.env` and
every `.env.*` file except `.env.example`, together with `.venv/`,
`__pycache__/`, `*.pyc`, `results/`, `results_export/`, `*.log` and
`.DS_Store`.

## Offline self-tests

Both self-tests need no key, make no network call and incur no charge. Each
replaces the provider client with a stub that returns a scripted tool call and
then a final message, while the rest of the path is the real AgentDojo code:
the real suites, attack, trace logger and scoring.

```sh
python src/selftest.py
python src/selftest_baseten.py
```

`src/selftest.py` covers the Gemini path: patch installation, pipeline
construction without AgentDojo's closed model enum, the attack's prose model
name, one task end to end with the trace written where the analysis expects
it, token and cost accounting including the long-context price tier, the
thinking configuration, resumption from the task cache, isolation between
worker threads, the negative-flip arithmetic on a hand-built case set, the
polarity of AgentDojo's `security` field, and the exclusion of crashed traces.

`src/selftest_baseten.py` covers the OpenAI-compatible path through the real
Baseten request shim: the message rewrites preserve every character, the
injected text carries no vendor-specific name, a temperature of 0 reaches the
wire, token accounting and pricing work, and the configuration hash under
`important_instructions_no_names` equals `28f36d30e3a053c1` while differing
from the lock.

Each script prints one line per check, followed by an overall verdict. The
Anthropic and OpenAI adapters have no offline self-test.

## Live pre-flight checks

These scripts list the models the key can serve, reconcile the list against
the registry, and make small generation and tool-calling requests. They call
the live API and therefore incur a small charge.

```sh
python src/verify_setup.py [--preset PRESET] [--skip-probes]    # google; default preset: all
python src/verify_setup_baseten.py [--preset PRESET]            # baseten; default preset: mvc
python src/verify_setup_anthropic.py [--preset PRESET]          # anthropic; default preset: sonnet-lineage
python src/verify_setup_openai.py                               # openai; no options, checks every registry model
```

`--skip-probes` on the Gemini check skips the generation and tool-call
requests but still lists the models the key can serve.

## Configuration freeze

```sh
python src/freeze.py [check]    # compare the current configuration with the lock (default)
python src/freeze.py write      # replace the lock
```

Use `check`. `write` replaces `config/harness.lock.json`, and results produced
under a new hash are not comparable with results produced under the shipped
one. The closing message of `setup.sh` still lists `python src/freeze.py write`
("freeze the harness configuration, once") among its next steps. That advice
predates the shipped lock: run `python src/freeze.py check` instead.

The hash does not cover the `execution` section of `harness.yaml`
(repetitions, concurrency, budget), the model registries, the suite subset
chosen with `--suites`, or anything on the provider's side. The manifest
records the suite subset and the registry separately.

### Configuration hashes used in the paper

| Hash | Attack | Configuration |
|---|---|---|
| `24a5898747a51439` | `important_instructions` | `config/harness.yaml` as shipped, and the hash stored in `config/harness.lock.json`. |
| `28f36d30e3a053c1` | `important_instructions_no_names` | The same configuration with `harness.attack` overridden, for example by `run_sweep.py --attack important_instructions_no_names --allow-config-change`. |

Both hashes share AgentDojo 0.1.35, benchmark v1.2.2, temperature 0.0, the
`as_deployed` thinking mode and no defense, and both can be recomputed with
`src/freeze.py` from this release. Overriding `--attack`, `--defense`,
`--thinking-mode` or `--temperature` produces a different hash. The OpenAI arm
refuses any temperature other than 1.0, and the Anthropic lineage includes a
model that accepts only 1.0, so runs of those arms at temperature 1.0 carry
hashes other than these two.

## Running a sweep

A sweep calls the provider's API and is billed at the provider's current
prices. Start with `--dry-run`, which prints the plan and a cost projection
and exits before any API call; it needs no key.

The command-line interface, as printed by `python src/run_sweep.py --help`:

```text
usage: run_sweep.py [-h] [--arm {anthropic,baseten,google,openai}] [--preset PRESET]
                    [--models MODELS [MODELS ...]] [--suites SUITES [SUITES ...]] [--reps REPS]
                    [--concurrency CONCURRENCY] [--budget-usd BUDGET_USD]
                    [--max-user-tasks MAX_USER_TASKS] [--run-id RUN_ID] [--defense DEFENSE]
                    [--attack ATTACK] [--thinking-mode {as_deployed,normalized,off}]
                    [--temperature TEMPERATURE] [--dry-run] [--allow-config-change]
```

- `--arm` defaults to `google`. One of `--preset` or `--models` is required;
  when both are given, `--models` is used. Without either, the script lists
  the arm's presets and exits with status 1. On the `anthropic` arm that
  listing instead fails with `KeyError: 'description'`, because the presets in
  `config/models_anthropic.yaml` carry no `description` field, so name the
  preset explicitly there.
- Presets: `google`: `smoke`, `flash-lite-lineage`, `flash-lineage`,
  `pro-lineage`, `all`. `baseten`: `smoke`, `kimi-lineage`, `glm-lineage`,
  `glm-serving`, `mvc`. `anthropic`: `sonnet-lineage`, `smoke`. `openai`:
  `smoke`, `flagship-pair`, `ladder-5-6`, `mvc`, `episode-b`,
  `three-lineages`.
- `--reps`, `--concurrency` and `--budget-usd` override the `execution`
  section of `harness.yaml` (3 repetitions, concurrency 4, 250.0 US dollars).
  The budget ceiling is enforced against measured token usage; cached tokens
  are priced at the full input rate, so every cost figure is an upper bound.
- `--defense` takes an AgentDojo defense name, or `none`.
- `--run-id` resumes an earlier run: tasks already present in AgentDojo's cache
  are skipped. An interrupt (Ctrl-C) finishes the tasks in flight and prints
  the resume command.

The smoke commands printed by the pre-flight scripts:

```sh
python src/run_sweep.py --preset smoke --suites slack --max-user-tasks 2 --reps 1
python src/run_sweep.py --arm baseten --preset smoke --suites slack \
    --max-user-tasks 2 --reps 1 --attack important_instructions_no_names \
    --allow-config-change
```

The OpenAI arm additionally needs `--temperature 1.0 --allow-config-change`,
and so does the Anthropic `sonnet-lineage` preset, because `claude-sonnet-5`
accepts only a temperature of 1.0.

A run writes `results/<arm>_<run_id>/`. The directory holds `manifest.json`
(configuration, effective hash, model aliases, cost and token totals) and, for
each model and repetition, AgentDojo's per-task trace files and a
`usage.jsonl` with the tokens and cost of every task.

`runners/` holds the two multi-run sequences used during the study
(`runners/README.md`). Both run `run_sweep.py` against live APIs and spend
money. `anthropic_sonnet.sh` passes a fixed `--run-id`, so re-invoking it
resumes; `flashlite5.sh` passes none, so each invocation starts new runs.

## Analysis

First export a run into the compact per-case form that most scripts read:

```sh
python src/export_results.py results/<arm>_<run_id>
```

This writes `results_export/<run_id>/` with `per_case.csv.gz` (one row per
model, repetition and case: `model`, `repetition`, `suite`, `user_task`,
`injection_task`, `attack`, `utility`, `secure`, `degenerate`, `duration_s`),
`models.csv`, `comparisons.json`, `summary.md` and `manifest.json`.

AgentDojo's `security` field is true when the attack succeeded. The export
inverts it once, so `secure = 1` means the attack failed. Repetitions are
collapsed by majority vote, `sum(votes) * 2 > len(votes)`, so a tie resolves to
insecure and, at two repetitions, a case must be secure in both.

Scripts that read a raw run directory, `results/<arm>_<run_id>`:

| Script | Usage | Purpose |
|---|---|---|
| `analyze.py` | `[--collapse {majority,strict}] [--attack ATTACK] [--json-out FILE] run_dir` | Per-model attack success rate, and the negative-flip rate between consecutive versions of each registry lineage, with a utility-matched view. |
| `compare_arms.py` | `[--json-out FILE] run_dir` | The open-weight arm against a Gemini export under the same name-neutral attack. Also reads `results_export/probe_no_names/per_case.csv.gz`, which is not shipped: export a Gemini run under `important_instructions_no_names` and name its export directory `probe_no_names`. |

Scripts that read an export directory, `results_export/<run_id>`:

| Script | Usage | Purpose |
|---|---|---|
| `arm_dashboard.py` | `[--json-out FILE] [--max-reps N] export_dir` | Per-model capability gate and every ordered pair, gated and ungated. |
| `noise_floor.py` | `[--suites [SUITE ...]] [--json-out FILE] export_dir` | Each model paired against itself across its repetitions. |
| `flip_anatomy.py` | `--before MODEL --after MODEL [--suites [SUITE ...]] [--gate {capability,raw}] [--json-out FILE] export_dir` | Negative flips split by whether each version solved the user task. |
| `degeneracy_check.py` | `--before MODEL --after MODEL [--suites [SUITE ...]] [--json-out FILE] export_dir` | The transition recomputed without cases where the earlier model produced an empty assistant turn. |
| `estimand_check.py` | `[--lineage MODEL [MODEL ...]] [--all-consecutive] [--max-reps N] [--json-out FILE] export_dir` | Transition flips and noise floor on the same per-repetition scale. `--lineage` may be repeated. |
| `gate_stability.py` | `[--pair BEFORE AFTER] [--json-out FILE] export_dir` | How far a capability-matched rate depends on the gate definition. |
| `utility_decomposition.py` | `[--json-out FILE] export_dir` | Clean utility split into user tasks and injection goals run standalone. |
| `compare_thinking_arms.py` | `--arm LABEL EXPORT_DIR [--arm LABEL EXPORT_DIR ...] --lineage MODEL [MODEL ...] [--suites [SUITE ...]] [--json-out FILE]` | Two exports of the same lineage, for example as-deployed against thinking-normalized, gated three ways. |
| `pool_exports.py` | `--out OUT_DIR [--allow-config-mismatch] export_dir [export_dir ...]` | Pools exports into one, and refuses when their hash, harness or generation settings, or suites differ. |
| `manuscript_numbers.py` | `[--arm {anthropic,crossvendor,gemini,gpt,openweight}]` | Recomputes the paper's tables from the exports named in its `ARMS` table. None of those exports ships here, so it reports each arm as missing until `ARMS` names your own exports. |

All scripts are run as `python src/<script> ...`, and each prints its full
option list with `--help`.

## What this release does not contain

- No API keys, credentials or `.env` file.
- No raw traces, no per-case results, no result exports or run manifests from
  the study, and no manuscript material or internal write-ups.

Rerunning the harness therefore requires your own provider accounts and keys,
and it is billed by the providers. The results will not be identical to the
study's. Provider-side sampling is not fully deterministic even at temperature
0, the Anthropic and OpenAI arms run at temperature 1.0, and hosted models,
snapshots and prices change or are withdrawn; each registry records the date
on which its identifiers and prices were verified, and an identifier that is no
longer served cannot be rerun.

## Integrity

`MANIFEST.sha256` lists the SHA-256 of every file except itself:

```sh
shasum -a 256 -c MANIFEST.sha256    # macOS
sha256sum -c MANIFEST.sha256        # Linux
```

## Citation

Please cite the accompanying paper: *When AI Systems Change: A Survey of
Security Reassessment across Inheritance, Updates, and Runtime Binding*.

## License

The harness is licensed under the Apache License 2.0; see `LICENSE` and
`NOTICE`. Every Python and shell source file carries the SPDX identifier
`Apache-2.0`. Portions adapted from AgentDojo 0.1.35 remain under the MIT
License; `THIRD_PARTY_NOTICES.md` names them and reproduces that license.
