#!/usr/bin/env python3
"""Build every task the release branch ships and report which ones fail.

The published benchmark tables pool results per task, so a variant that cannot be
constructed from the release distribution is a broken promise even when the number in the
table is real. This walks the registry, builds each env, resets it and takes a step.
"""

from __future__ import annotations

import os
import sys
import traceback

import gs_gym


def main() -> int:
    subset = [t for t in os.environ.get("TASKS", "").split() if t]
    tasks = subset or sorted(gs_gym.list_tasks())
    print(f"building {len(tasks)} task(s)\n")

    failures: dict[str, str] = {}
    for task in tasks:
        try:
            env = gs_gym.make(task, n_envs=1)
            env.reset()
            env.step(env.action_space.sample())
            env.close()
            print(f"OK   {task}", flush=True)
        except Exception as exc:  # noqa: BLE001 - we want every failure, not the first
            failures[task] = f"{type(exc).__name__}: {exc}"
            print(f"FAIL {task}: {failures[task].splitlines()[0][:150]}", flush=True)
            traceback.clear_frames(sys.exc_info()[2])

    print(f"\n{len(tasks) - len(failures)}/{len(tasks)} built")
    if failures:
        print("\nfailures:")
        for task, err in sorted(failures.items()):
            print(f"  {task}\n      {err[:200]}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
