#!/usr/bin/env python3
"""Summarize lerobot-eval results under a run directory.

Usage:
    python summarize_results.py <run_dir> [<run_dir> ...]

Each run_dir should contain subdirectories (one per task suite), each with an
eval_info.json produced by lerobot-eval. Writes summary.txt into each run_dir.

Output columns: task_id | seed SR | seed PS | mut SR | mut PS
Rows: one per RoboWits task number. robowits_10 tasks appear first.

This module also owns the shared eval bookkeeping the VLM baseline scripts need:
the suite/mutation-variant definitions, and the `eval_info.json` writers. Those
live next to the reader here so the two cannot drift apart.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Task suites
# ──────────────────────────────────────────────────────────────────────────────

MUTATION_RE = re.compile(r"^robowits/(\d\d)_(\d+)-v0$")
# A per-variant results directory, e.g. "02_01" inside "mutation_02".
VARIANT_DIR_RE = re.compile(r"^\d\d_\d+$")

# The base tasks of the robowits_10 suite.
ROBOWITS_10 = ["01", "02", "03", "04", "06", "09", "13", "16", "17", "25"]

# Every RoboWits task.
ROBOWITS_ALL = [f"{i:02d}" for i in range(1, 31)]

# The 20 tasks outside robowits_10.
ROBOWITS_20 = [t for t in ROBOWITS_ALL if t not in ROBOWITS_10]

# Tasks with SPH/MPM particle or deformable physics. These are far slower than the rest
# (task 15 alone stalled a CPU-backend sweep for hours) and get halved episode counts.
# This list only defines the "slow" suite: which backend a task runs on is declared by the
# env class itself (GenesisGymEnv.preferred_genesis_backend), which also covers mutation
# variants that add particles to an otherwise rigid task.
SLOW_TASKS = ["08", "12", "15", "20", "21", "22", "25", "30"]

# The remaining 22 tasks, fast enough for the CPU backend.
NORMAL_TASKS = [t for t in ROBOWITS_ALL if t not in SLOW_TASKS]

# Mutation variants excluded from evaluation. Variants with no eval dataset are *not* listed
# here -- eval_robowits_10.sh skips variants with no eval dataset, so those drop out on their own,
# and the SPH-liquid variants of task 02 no longer need excluding either: each env declares its
# own preferred_genesis_backend, so they are simply routed to the CUDA backend. Only judgement
# calls belong here, and there are none left: 02_03 ("extra distractors") was the last, and that
# call has been reversed.
EXCLUDED_MUTATIONS: list[str] = []

# Suite name -> base task ids. "all" is every task, not just robowits_10.
SUITES: dict[str, list[str]] = {
    "robowits_10": ROBOWITS_10,
    "robowits_20": ROBOWITS_20,
    "normal": NORMAL_TASKS,
    "slow": SLOW_TASKS,
    "all": ROBOWITS_ALL,
}

# The suites `lerobot-eval` itself can produce, mirroring NAMED_SUITES in
# third_party/lerobot/src/lerobot/envs/factory.py. Deliberately *not* an alias
# for SUITES: a results directory is only index-mapped when lerobot could have
# written it, and lerobot knows only robowits_10.
NAMED_SUITES: dict[str, list[str]] = {"robowits_10": ROBOWITS_10}

ROBOWITS_10_ORDER = ROBOWITS_10


def resolve_suite(spec: str) -> list[str]:
    """Resolve a suite name or a single task id to a list of base task ids."""
    if spec in SUITES:
        return SUITES[spec]
    padded = spec.zfill(2)
    if padded in ROBOWITS_ALL:
        return [padded]
    raise ValueError(f"Unknown suite or task id: '{spec}' (expected one of {sorted(SUITES)} or a task id)")


# Task metadata: name, instruction, category and the full variant list per base task. This is
# the authoritative list of which mutation variants exist -- the task registry only knows what
# happens to be imported, and the eval dataset directory only knows what has scenes recorded.
METADATA_JSON = Path(__file__).resolve().parent.parent / "helpers" / "metadata.json"


def metadata_variants(task_id: str, metadata_path: str | Path = METADATA_JSON) -> list[str]:
    """Mutation variant stems for a base task, in metadata order: "02" -> ["02_01", ...].

    The seed entry (keyed by its module name, e.g. "02_retrieve_cube") is skipped; only the
    "mutation/<stem>" entries are returned.
    """
    with open(metadata_path) as f:
        meta = json.load(f)
    entry = meta.get(task_id.zfill(2))
    if entry is None:
        return []
    return [k.split("/", 1)[1] for k in entry.get("variants", {}) if k.startswith("mutation/")]


def mutation_variants(task_id: str) -> list[str]:
    """All registered mutation task names for a base task id, sorted.

    `gs_gym` is imported lazily: summarizing results is otherwise a pure-stdlib
    operation, and importing the task registry drags in Genesis.
    """
    from gs_gym.envs.registry import TASK_REGISTRY

    out = []
    for name in TASK_REGISTRY:
        m = MUTATION_RE.match(name)
        if m and m.group(1) == task_id.zfill(2):
            out.append(name)
    return sorted(out)


def resolve_task(task: str) -> str:
    """Resolve a seed task id ("02", "2"), or a full gs-gym task name, to a registered name.

    Mutations ("robowits/02_03-v0") have no short id and are returned unchanged.
    """
    import gs_gym  # noqa: F401  (registers the tasks)
    from gs_gym.envs.registry import TASK_REGISTRY

    if task.isdigit():
        prefix = f"robowits/{task.zfill(2)}-"
        for name in TASK_REGISTRY:
            if name.startswith(prefix):
                return name
    return task


def variant_stem(task_name: str) -> str:
    """Short form of a mutation task name: "robowits/02_03-v0" -> "02_03".

    This is the name used for both the variant's results directory and its eval
    dataset file.
    """
    return task_name.split("/", 1)[1].removesuffix("-v0")


def dataset_file(task_name: str, eval_json_dir: str | Path) -> Path:
    """Path to the eval dataset JSON for a mutation task name.

    Mutations are registered as `robowits/<task>_<variant>-v0` (e.g.
    "robowits/02_03-v0") and their per-episode initial placements live in
    `dataset/robowits/eval_dataset_mutation/<task>_<variant>.json`.
    """
    return Path(eval_json_dir) / f"{variant_stem(task_name)}.json"


def variants_with_dataset(task_id: str, eval_json_dir: str | Path) -> tuple[list[str], list[str]]:
    """Split a base task's mutation variants by whether eval data exists.

    Variants without a dataset file would fall back to random placement, which
    would not be comparable with the policy evals, so callers should skip them
    and report them rather than running them silently.

    Returns:
        (variants_with_data, variants_without_data)
    """
    have, missing = [], []
    for name in mutation_variants(task_id):
        (have if dataset_file(name, eval_json_dir).exists() else missing).append(name)
    return have, missing


# ──────────────────────────────────────────────────────────────────────────────
# Writing eval_info.json
#
# The VLM baselines do not run through `lerobot-eval`, but their results should
# land in the same tables as the policy evals. summarize_run() below reads one
# eval_info.json per task directory and prefers `per_task[0].metrics` for
# episode-level data, falling back to `overall`.
# ──────────────────────────────────────────────────────────────────────────────


def write_eval_info(
    output_dir: Path,
    task_name: str,
    successes: Sequence[bool],
    max_rewards: Sequence[float] | None = None,
) -> Path:
    """Write eval_info.json for one task's results.

    Args:
        output_dir: The task's results directory (the name summarize_results.py
            keys on, e.g. ".../02").
        task_name: Registered gs-gym task name, recorded for traceability.
        successes: Per-episode success flags.
        max_rewards: Per-episode progress score (max milestone reward reached,
            the same quantity lerobot-eval reports as `max_reward`). Omitting it
            falls back to the success flags, which is only meaningful for a
            pass/fail task -- callers that can measure progress should pass it.

    Returns:
        Path to the written file.
    """
    successes_int = [int(bool(s)) for s in successes]
    rewards = list(max_rewards) if max_rewards is not None else [float(s) for s in successes_int]
    n = len(successes_int)

    info = {
        "task_name": task_name,
        "per_task": [
            {
                "task_id": 0,
                "metrics": {
                    "successes": successes_int,
                    "max_rewards": rewards,
                },
            }
        ],
        "overall": {
            "pc_success": 100.0 * sum(successes_int) / n if n else 0.0,
            "avg_max_reward": sum(rewards) / n if n else 0.0,
            "n_episodes": n,
        },
    }
    return _dump_eval_info(output_dir, info)


def write_grouped_eval_info(
    output_dir: Path,
    group_name: str,
    entries: Sequence[tuple[str, Sequence[bool], Sequence[float] | None]],
) -> Path:
    """Write one eval_info.json aggregating several sub-tasks.

    Used for mutation suites: the `mutation_XX` branch of summarize_run() pools
    `per_task[].metrics.successes` across every variant of a base task, so all
    variants of e.g. task 02 belong in a single `mutation_02/eval_info.json`.

    Args:
        output_dir: Directory to write into (e.g. ".../mutation_02").
        group_name: Name recorded for the group (e.g. "robowits_mutation_02").
        entries: (task_name, successes, max_rewards) per variant, in a stable
            order. `max_rewards` may be None to fall back to the success flags.

    Returns:
        Path to the written file.
    """
    per_task = []
    all_successes: list[int] = []
    all_rewards: list[float] = []

    for idx, (name, successes, rewards_in) in enumerate(entries):
        successes_int = [int(bool(s)) for s in successes]
        rewards = [float(r) for r in rewards_in] if rewards_in is not None else [float(s) for s in successes_int]
        per_task.append(
            {
                "task_id": idx,
                "task_name": name,
                "metrics": {"successes": successes_int, "max_rewards": rewards},
            }
        )
        all_successes.extend(successes_int)
        all_rewards.extend(rewards)

    n = len(all_successes)
    info = {
        "task_name": group_name,
        "per_task": per_task,
        "overall": {
            "pc_success": 100.0 * sum(all_successes) / n if n else 0.0,
            "avg_max_reward": sum(all_rewards) / n if n else 0.0,
            "n_episodes": n,
        },
    }
    return _dump_eval_info(output_dir, info)


def _dump_eval_info(output_dir: Path, info: dict) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "eval_info.json"
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    return path


def load_eval_info(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def per_task_variant(entry: dict) -> str | None:
    """Variant directory name for one `per_task` entry, or None if it has no variant.

    Two writers produce these entries and they name variants differently:
    `lerobot-eval` splits the variant across `task_group` ("02") and `task_id`
    ("01"), while write_grouped_eval_info() records the full registered
    `task_name` ("robowits/02_01-v0") and uses `task_id` as a plain index.
    """
    if "task_group" in entry:
        return f"{entry['task_group']}_{entry['task_id']}"
    task_name = entry.get("task_name")
    return variant_stem(task_name) if task_name and "/" in task_name else None


def variant_ran(variant_dir: str) -> bool:
    """True if a mutation variant produced real rollouts (videos or parquet data).

    Variants whose env crashed on reset (e.g. stale MPM particle data in the eval
    scene) still leave five all-zero episodes in eval_task_info.json, and the
    resume logic in lerobot_eval keeps replaying those zeros on every re-run.
    Pooling them drags the task's score toward 0, so they are dropped here.

    A missing variant directory means we cannot tell (some suites use different
    layouts), so the variant is kept.
    """
    recordings = os.path.join(variant_dir, "recordings")
    if not os.path.isdir(recordings):
        return True
    for sub, ext in (("videos", ".mp4"), ("data", ".parquet")):
        for _, _, files in os.walk(os.path.join(recordings, sub)):
            if any(f.endswith(ext) for f in files):
                return True
    return False


def pool_mutation_results(run_dir: str | Path) -> None:
    """Pool per-variant eval_info.json files into their `mutation_XX/eval_info.json`.

    Evaluating one task id at a time (which is what letting each task pick its own Genesis
    backend requires) writes `mutation_XX/<variant>/eval_info.json`, but the `mutation_XX`
    branch of summarize_run() reads one pooled file per base task. The pooled file is derived
    data, so summarize_run() rebuilds it on the way in rather than making callers remember a
    separate step. Directories written by lerobot's own mutation suite hold no variant-stem
    subdirectories and are left untouched.
    """
    for mut_dir in sorted(Path(run_dir).glob("mutation_*")):
        if not mut_dir.is_dir():
            continue
        entries = []
        for var_dir in sorted(d for d in mut_dir.iterdir() if d.is_dir()):
            if not VARIANT_DIR_RE.match(var_dir.name):
                continue
            info_path = var_dir / "eval_info.json"
            if not info_path.exists():
                print(f"[warn] {mut_dir.name}: {var_dir.name} has no eval_info.json, skipping")
                continue
            per_task = load_eval_info(str(info_path)).get("per_task", [])
            metrics = per_task[0]["metrics"] if per_task else {}
            successes = metrics.get("successes", [])
            if not successes:
                print(f"[warn] {mut_dir.name}: {var_dir.name} has no episodes, skipping")
                continue
            entries.append((f"robowits/{var_dir.name}-v0", successes, metrics.get("max_rewards")))
        if entries:
            base = mut_dir.name.removeprefix("mutation_")
            write_grouped_eval_info(mut_dir, f"robowits_mutation_{base}", entries)
            n = sum(len(e[1]) for e in entries)
            print(f"Pooled {len(entries)} variant(s), {n} episode(s) -> {mut_dir}/eval_info.json")


def summarize_run(run_dir: str) -> None:
    pool_mutation_results(run_dir)
    task_dirs = sorted(d for d in os.listdir(run_dir) if os.path.isfile(os.path.join(run_dir, d, "eval_info.json")))
    if not task_dirs:
        print(f"[warn] no eval_info.json found under {run_dir}")
        return

    seed_tasks: dict[str, dict] = {}  # task_num -> {sr, ps, n}
    mut_tasks: dict[str, dict] = {}  # task_num -> {sr, ps, n}

    for td in task_dirs:
        info = load_eval_info(os.path.join(run_dir, td, "eval_info.json"))

        # Mutation suite: mutation_XX or mut_XX (skip old_*)
        mut_match = re.match(r"^(?:mutation|mut)_(\d+)$", td)
        if mut_match:
            task_num = mut_match.group(1).zfill(2)
            all_successes, all_max_rewards = [], []
            skipped = []
            for t in info.get("per_task", []):
                variant = per_task_variant(t)
                if variant and not variant_ran(os.path.join(run_dir, td, variant)):
                    skipped.append(variant)
                    continue
                m = t["metrics"]
                all_successes.extend(m.get("successes", []))
                all_max_rewards.extend(m.get("max_rewards", []))
            if skipped:
                print(f"[warn] {td}: dropping {len(skipped)} variant(s) with no rollouts: {', '.join(skipped)}")
            if all_successes:
                sr = 100.0 * sum(all_successes) / len(all_successes)
                ps = sum(all_max_rewards) / len(all_max_rewards) if all_max_rewards else 0.0
                n = len(all_successes)
            elif skipped:
                # Every variant was dropped -- report N/A rather than the polluted overall.
                sr = ps = float("nan")
                n = 0
            else:
                ov = info.get("overall", {})
                sr = ov.get("pc_success", 0.0)
                ps = ov.get("avg_max_reward", 0.0)
                n = ov.get("n_episodes", 0)
            mut_tasks[task_num] = {"sr": sr, "ps": ps, "n": n}
            continue

        # Named seed suite (e.g. robowits_10): map per_task index -> task_num
        if td in NAMED_SUITES:
            id_map = NAMED_SUITES[td]
            for t in info.get("per_task", []):
                idx = t["task_id"]
                task_num = id_map[idx] if idx < len(id_map) else str(idx).zfill(2)
                m = t["metrics"]
                successes = m.get("successes", [])
                max_rewards = m.get("max_rewards", [])
                sr = 100.0 * sum(successes) / len(successes) if successes else 0.0
                ps = sum(max_rewards) / len(max_rewards) if max_rewards else 0.0
                seed_tasks[task_num] = {"sr": sr, "ps": ps, "n": len(successes)}
            continue

        # Single-task seed suite: directory name is the task number (e.g. "01", "02")
        if re.match(r"^\d+$", td):
            task_num = td.zfill(2)
            ov = info.get("overall", {})
            # prefer per_task[0] for episode-level data
            pt = info.get("per_task", [])
            if pt:
                m = pt[0]["metrics"]
                successes = m.get("successes", [])
                max_rewards = m.get("max_rewards", [])
                sr = 100.0 * sum(successes) / len(successes) if successes else 0.0
                ps = sum(max_rewards) / len(max_rewards) if max_rewards else 0.0
                n = len(successes)
            else:
                sr = ov.get("pc_success", 0.0)
                ps = ov.get("avg_max_reward", 0.0)
                n = ov.get("n_episodes", 0)
            seed_tasks[task_num] = {"sr": sr, "ps": ps, "n": n}
            continue

        # Skip unrecognised directories (e.g. old_mutation_*)

    all_task_nums = sorted(set(list(seed_tasks.keys()) + list(mut_tasks.keys())))
    ordered = [t for t in ROBOWITS_10_ORDER if t in all_task_nums]
    ordered += [t for t in all_task_nums if t not in ordered]

    NAN = float("nan")
    MISSING = {"sr": NAN, "ps": NAN, "n": 0}
    rows = [{"task_id": t, "seed": seed_tasks.get(t, MISSING), "mut": mut_tasks.get(t, MISSING)} for t in ordered]

    def safe_mean(vals):
        vs = [v for v in vals if v == v]
        return sum(vs) / len(vs) if vs else NAN

    overall_seed_sr = safe_mean([r["seed"]["sr"] for r in rows])
    overall_seed_ps = safe_mean([r["seed"]["ps"] for r in rows])
    overall_mut_sr = safe_mean([r["mut"]["sr"] for r in rows])
    overall_mut_ps = safe_mean([r["mut"]["ps"] for r in rows])

    rw10_ids = set(ROBOWITS_10_ORDER)
    rw10_rows = [r for r in rows if r["task_id"] in rw10_ids]
    other_rows = [r for r in rows if r["task_id"] not in rw10_ids]

    rw10_seed_sr = safe_mean([r["seed"]["sr"] for r in rw10_rows])
    rw10_seed_ps = safe_mean([r["seed"]["ps"] for r in rw10_rows])
    rw10_mut_sr = safe_mean([r["mut"]["sr"] for r in rw10_rows])
    rw10_mut_ps = safe_mean([r["mut"]["ps"] for r in rw10_rows])

    col_id = max(max((len(r["task_id"]) for r in rows), default=0), 7) + 2

    def fmt_sr(v):
        return f"{v:>8.1f}%" if v == v else f"{'N/A':>9}"

    def fmt_ps(v):
        return f"{v:>9.4f}" if v == v else f"{'N/A':>9}"

    header = f"{'task_id':<{col_id}} {'seed SR':>9} {'seed PS':>9} {'mut SR':>9} {'mut PS':>9}"
    sep = "-" * len(header)

    lines = [
        f"Run: {os.path.abspath(run_dir)}",
        f"Tasks: {len(rows)}  (seed: {len(seed_tasks)}  mut: {len(mut_tasks)})",
        "",
        header,
        sep,
        f"{'RW10 AVG':<{col_id}} {fmt_sr(rw10_seed_sr)} {fmt_ps(rw10_seed_ps)}"
        f" {fmt_sr(rw10_mut_sr)} {fmt_ps(rw10_mut_ps)}",
        f"{'OVERALL':<{col_id}} {fmt_sr(overall_seed_sr)} {fmt_ps(overall_seed_ps)}"
        f" {fmt_sr(overall_mut_sr)} {fmt_ps(overall_mut_ps)}",
        sep,
    ]

    for r in rw10_rows:
        lines.append(
            f"{r['task_id']:<{col_id}} {fmt_sr(r['seed']['sr'])} {fmt_ps(r['seed']['ps'])}"
            f" {fmt_sr(r['mut']['sr'])} {fmt_ps(r['mut']['ps'])}"
        )

    if other_rows:
        lines.append(sep)
        for r in other_rows:
            lines.append(
                f"{r['task_id']:<{col_id}} {fmt_sr(r['seed']['sr'])} {fmt_ps(r['seed']['ps'])}"
                f" {fmt_sr(r['mut']['sr'])} {fmt_ps(r['mut']['ps'])}"
            )

    output = "\n".join(lines) + "\n"
    out_path = os.path.join(run_dir, "summary.txt")
    with open(out_path, "w") as f:
        f.write(output)
    print(f"Wrote {out_path}")
    print(
        f"  {len(rows)} tasks  |"
        f"  rw10 seed SR: {rw10_seed_sr:.1f}%  rw10 mut SR: {rw10_mut_sr:.1f}%  |"
        f"  overall seed SR: {overall_seed_sr:.1f}%  overall mut SR: {overall_mut_sr:.1f}%"
    )


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for run_dir in sys.argv[1:]:
        summarize_run(run_dir)


if __name__ == "__main__":
    main()
