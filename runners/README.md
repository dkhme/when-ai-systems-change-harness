# Runner scripts

The exact multi-run sequences behind the study's results, which are not part of
this release. Kept here rather than in a scratch directory because a results
file without the command that produced it is not reproducible, and scratch
directories do not survive the session that made them.

| Script | Produces |
|---|---|
| `flashlite5.sh` | The five-experiment Flash-Lite hardening campaign of 2026-08-09. |
| `anthropic_sonnet.sh` | The Sonnet arm, both rounds, under `results/anthropic_20260810T044952Z`. |

Single runs are started directly with `src/run_sweep.py`; only multi-run
sequences live here.

**Neither script should be re-run wholesale.** AgentDojo caches every completed
task under its run directory, so a sweep resumed with its `--run-id` skips
completed work and pays only for what is missing. `anthropic_sonnet.sh` passes a
fixed `--run-id`, so re-invoking the script resumes it. `flashlite5.sh` passes
no `--run-id`, so every invocation starts new run directories and pays again for
every item; to resume an interrupted item, re-run that item's `run_sweep.py`
command with `--run-id` set to the id in the `manifest` path it printed.
