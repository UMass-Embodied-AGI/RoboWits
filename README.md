<p align="center">
  <h1 align="center">RoboWits: Unexpected Challenges for Robotic Creative Problem Solving</h1>
  <p align="center">
    NeurIPS 2026 Evaluations &amp; Datasets Track
  </p>
  <p align="center">
    <a href="https://chunru-lin.github.io/">Chunru Lin</a><sup>*</sup>,
    <a href="https://icefoxzhx.github.io/">Hongxin Zhang</a><sup>*</sup>,
    <a href="https://ario-photography.netlify.app/dist/about_me">Fenghao Yu</a>,
    <a href="https://acmlczh.github.io/">Zhehuan Chen</a>,
    <a href="https://cocosci.princeton.edu/tom/index.php">Thomas L. Griffiths</a>,
    <a href="https://yejinc.github.io/">Yejin Choi</a>,
    <a href="https://davheld.github.io/">David Held</a>
    <a href="https://people.csail.mit.edu/ganchuang/">Chuang Gan</a>
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/2605.30326">
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://umass-embodied-agi.github.io/RoboWits/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
    <a href='https://huggingface.co/datasets/XHRlyb2001/RoboWits_lerobot_dataset' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Dataset-RLBench-orange?style=flat&logo=Hugging%20face&logoColor=orange' alt='Dataset Hugging Face'>
    </a>
  </p>
</p>

**RoboWits**, a bi-manual robotic benchmark designed to systematically evaluate cognitive reasoning, creative tool use, and robustness to unexpected conditions.

<p align="center">
    <img src="assets/teaser.png" alt="Logo" width="190%">
</p>

<br>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#news">News</a>
    </li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#quick-start">Quick Start</a>
    </li>
    <li>
      <a href="#benchmark">Benchmark</a>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>

## News
- [2026-09-27] RoboWits now runs on Genesis World 1.3.1, with its latest physics and rendering. The latest full benchmark is updated now.
- [2026-09-24] RoboWits is accepted to the **NeurIPS 2026 Evaluations & Datasets Track**!
- [2026-06-04] We release the Benchmark code, along with the [dataset](https://huggingface.co/datasets/XHRlyb2001/RoboWits_lerobot_dataset) consisting of ~50 demonstrations on 24 seed tasks for fine-tuning!
- [2026-05-30] RoboWits is on [arXiv](https://arxiv.org/abs/2605.30326)! Check out our [project website](https://umass-embodied-agi.github.io/RoboWits/) for videos.


## Installation

### Dependencies

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already.

```bash
# Fetch the patched LeRobot submodule (third_party/lerobot)
git submodule update --init --recursive

uv sync
source .venv/bin/activate
```

### Assets

Some assets come from [BlenderKit](https://www.blenderkit.com/) and require an API key to download, which can be found in your [profile](https://www.blenderkit.com/profile/addon/). Some BlenderKit assets are only available in `.blend` format — the download script converts them to GLB automatically by invoking Blender as a subprocess.

#### Install Blender

Make sure Blender is installed and available on your `PATH`:

```bash
# macOS (Homebrew)
brew install --cask blender

# Or download from https://www.blender.org/download/ and add to PATH
echo 'export PATH="/Applications/Blender.app/Contents/MacOS:$PATH"' >> ~/.zshrc && source ~/.zshrc

# Ubuntu (headless) — download the latest tarball from https://www.blender.org/download/
wget https://mirrors.dotsrc.org/blender/release/Blender5.1/blender-5.1.2-linux-x64.tar.xz
tar -xf blender-5.1.2-linux-x64.tar.xz -C /opt
echo 'export PATH="/opt/blender-5.1.2-linux-x64:$PATH"' >> ~/.bashrc && source ~/.bashrc

# Verify
blender --version
```

#### Prepare Assets

```bash
bash assets/setup_assets.sh --api-key <BLENDERKIT_API_KEY>
```

> **Note:** Some assets require a BlenderKit full-plan subscription. If your API key is a free-tier key, downloads for paid assets will be skipped automatically and those tasks will be unavailable.

The directory should look like:

```
assets/
  hf_assets/
    work_table.glb
    marvin_bimanual/
      ...
    worktable_texture/
      grained black plastic_Normal.jpg
      grained black plastic_Roughness.jpg
    ...
  blender_kit/
    <asset-id>/
      obj.glb
    ...
```

## Quick Start

### Run the environment

```bash
python scripts/robowits/examples/run_env.py 
```

### Available Tasks

```python
import gs_gym

# List all tasks
print(gs_gym.list_tasks())

# List tasks by benchmark
print(gs_gym.list_tasks(benchmark="robowits"))

# List all benchmarks
print(gs_gym.list_benchmarks())
```

### Observation Modes

Configure via `observation_mode` parameter:

| Mode | Dim | Contents |
|------|-----|----------|
| `"EE"` (default) | 14D | Right EE pos (3) + left EE pos (3) + right rot axis-angle (3) + left rot axis-angle (3) + grippers (2) |
| `"EE_ROT6D"` | 20D | Right EE pos (3) + left EE pos (3) + right rot6d (6) + left rot6d (6) + grippers (2) |
| `"JOINT"` | 16D | Right joints (7) + right gripper (1) + left joints (7) + left gripper (1) |

`rot6d` is the first two **rows** of the rotation matrix flattened row-major,
`[R00, R01, R02, R10, R11, R12]` — a continuous parameterization that avoids the discontinuity
axis-angle has near ±π. EE positions are robot-base-relative in every EE mode.

### Control Modes

Configure via `control_mode` parameter:

| Mode | Dim | Description |
|------|-----|-------------|
| `"EE_ABS"` (default) | 14D | Absolute end-effector pose, axis-angle rotation (IK handled internally) |
| `"EE_ABS_ROT6D"` | 20D | Absolute end-effector pose, rot6d rotation (IK handled internally) |
| `"EE_DELTA"` | 14D | Delta end-effector pose |
| `"JOINT_ABS"` | 16D | Absolute joint positions |
| `"JOINT_DELTA"` | 16D | Delta joint positions (gripper width is absolute) |

Both EE modes lay the action out as `[R_pos(3), L_pos(3), R_rot(3 or 6), L_rot(3 or 6), R_grip(1), L_grip(1)]`.

### Observation Format

**Environment output:**
- `"agent_pos"`: (n_envs, D) numpy array — robot state for the policy; D depends on `observation_mode`
- `"agent_pos_joint"`: (n_envs, 16) numpy array — JOINT state, always present regardless of `observation_mode`
- `"agent_pos_ee"`: (n_envs, 14) numpy array — EE state, always present regardless of `observation_mode`
- `"agent_pos_ee_rot6d"`: (n_envs, 20) numpy array — EE state with rot6d rotations, always present regardless of `observation_mode`
- `"pixels"`: dict of (n_envs, H, W, 3) numpy uint8 arrays — camera images
  - `"ego"`: Static ego-view camera
  - `"wrist_right"`: Right wrist camera
  - `"wrist_left"`: Left wrist camera

**After `preprocess_observation()` + `add_envs_task()`** (LeRobot format):
- `"observation.state"`: torch.Tensor (n_envs, D) — policy input, mirrors `agent_pos`
- `"observation.agent_pos_joint"`: torch.Tensor (n_envs, 16) — always recorded
- `"observation.agent_pos_ee"`: torch.Tensor (n_envs, 14) — always recorded
- `"observation.images.ego"`: torch.Tensor (n_envs, 3, H, W) — channel-first, normalized [0, 1]
- `"observation.images.wrist_right"`: torch.Tensor (n_envs, 3, H, W)
- `"observation.images.wrist_left"`: torch.Tensor (n_envs, 3, H, W)
- `"task"`: list[str] (n_envs,) — natural language task description, added by lerobot-eval

### Training and Evaluation

RoboWits integrates with LeRobot via a patched fork tracked as a git submodule at `third_party/lerobot/` (see [Installation](#installation)). All scripts use `lerobot-train` / `lerobot-eval` and accept configuration via environment variables.

#### Training

A dataset containing ~50 human demonstrations for ~24 robowits seed tasks is available on HuggingFace at [`XHRlyb2001/RoboWits_lerobot_dataset`](https://huggingface.co/datasets/XHRlyb2001/RoboWits_lerobot_dataset). Here are some example training script with the dataset.

```bash
# ACT (env vars: HF_DATASET, OUTPUT_DIR, NUM_PROCESSES, STEPS, SAVE_FREQ, VAL_FREQ)
bash scripts/robowits/train/train_act.sh

# Pi0
bash scripts/robowits/train/train_pi0.sh

# Pi0.5
bash scripts/robowits/train/train_pi05.sh
```

All training scripts use `accelerate launch --multi_gpu` with W&B logging enabled by default. Checkpoints are saved to `checkpoints/<policy>_robowits/` by default.

#### Evaluation

```bash
# Evaluate on robowits-10 seed tasks (env vars: CHECKPOINT_PATH, TASK_IDS, CONFIG, OBSERVATION_MODE, N_EPISODES)
CHECKPOINT_PATH=/path/to/checkpoint TASK_IDS="01 02 03 04 06 09 13 16 17 25" bash scripts/robowits/eval/eval.sh

# Evaluate on mutation tasks
CHECKPOINT_PATH=/path/to/checkpoint TASK_IDS="01 02 03 04 06 09 13 16 17 25" bash scripts/robowits/eval/eval_mutation.sh
```

`CONFIG` selects the control mode and `OBSERVATION_MODE` the observation mode; they default to
`EE_ABS` and `EE`. Match them to what the policy was trained on — a checkpoint trained on rot6d
needs both:

```bash
CONFIG=EE_ABS_ROT6D OBSERVATION_MODE=EE_ROT6D CHECKPOINT_PATH=/path/to/checkpoint \
    bash scripts/robowits/eval/eval.sh
```

## Benchmark

Success rate and average progress score over 50 episodes for seed and 20 episodes for mutations are
reported.

### RoboWits-10

| Task               |    ACT    |           |    Pi0    |           |   Pi0.5   |           |
|--------------------|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
|                    |   Seed    |    Mut    |   Seed    |    Mut    |   Seed    |    Mut    |
| 01 Align Blocks    |14.0%, 0.61|10.0%, 0.58|14.0%, 0.63|20.8%, 0.62|20.0%, 0.63|15.0%, 0.63|
| 02 Retrieve Cube   | 2.0%, 0.29| 0.0%, 0.32| 6.0%, 0.30| 0.8%, 0.31| 0.0%, 0.28| 1.7%, 0.32|
| 03 Gap Retrieve    | 2.0%, 0.51| 0.0%, 0.43|18.0%, 0.65| 5.0%, 0.46|20.0%, 0.66| 6.0%, 0.49|
| 04 Pinch Card      | 0.0%, 0.41| 0.0%, 0.53|10.0%, 0.63| 9.0%, 0.62| 2.0%, 0.54| 7.0%, 0.59|
| 06 Dominos         |52.0%, 0.87|38.3%, 0.86|86.0%, 0.97|57.5%, 0.87|74.0%, 0.95|52.5%, 0.84|
| 09 Hold Cup        | 0.0%, 0.53| 0.0%, 0.39| 0.0%, 0.66| 0.0%, 0.45| 2.0%, 0.64| 0.0%, 0.46|
| 13 Cover With Lid  | 0.0%, 0.48| 0.0%, 0.37| 0.0%, 0.53| 0.0%, 0.43| 0.0%, 0.58| 0.0%, 0.48|
| 16 Stand Bulb      | 0.0%, 0.54| 0.0%, 0.41| 0.0%, 0.55| 0.0%, 0.42| 0.0%, 0.55| 0.0%, 0.42|
| 17 Ball Onto Tower | 0.0%, 0.55| 0.0%, 0.48| 0.0%, 0.62| 0.0%, 0.54| 0.0%, 0.67| 0.0%, 0.58|
| 25 Water Into Mug  | 2.0%, 0.36| 0.7%, 0.43| 0.0%, 0.36| 0.0%, 0.45| 4.0%, 0.35| 0.0%, 0.47|
| **Average**        | 7.2%, 0.52| 4.9%, 0.48|13.4%, 0.59| 9.3%, 0.52|12.2%, 0.59| 8.2%, 0.53|

### RoboWits-Full

| Task                        |    ACT    |           |    Pi0    |           |   Pi0.5   |           |
|-----------------------------|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
|                             |   Seed    |    Mut    |   Seed    |    Mut    |   Seed    |    Mut    |
| 01 Align Blocks             |14.0%, 0.61|10.0%, 0.58|14.0%, 0.63|20.8%, 0.62|20.0%, 0.63|15.0%, 0.63|
| 02 Retrieve Cube            | 2.0%, 0.29| 0.0%, 0.32| 6.0%, 0.30| 0.8%, 0.31| 0.0%, 0.28| 1.7%, 0.32|
| 03 Gap Retrieve             | 2.0%, 0.51| 0.0%, 0.43|18.0%, 0.65| 5.0%, 0.46|20.0%, 0.66| 6.0%, 0.49|
| 04 Pinch Card               | 0.0%, 0.41| 0.0%, 0.53|10.0%, 0.63| 9.0%, 0.62| 2.0%, 0.54| 7.0%, 0.59|
| 05 Roll Up Ball             | 0.0%, 0.36| 0.0%, 0.25| 0.0%, 0.42| 0.0%, 0.31| 0.0%, 0.50| 0.0%, 0.31|
| 06 Dominos                  |52.0%, 0.87|38.3%, 0.86|86.0%, 0.97|57.5%, 0.87|74.0%, 0.95|52.5%, 0.84|
| 07 Stand Pages              | 0.0%, 0.24| 0.0%, 0.16| 0.0%, 0.29| 0.0%, 0.29| 0.0%, 0.29| 0.0%, 0.13|
| 08 Round Dough Sheet        | 0.0%, 0.52| 0.0%, 0.52| 0.0%, 0.58| 0.0%, 0.58| 0.0%, 0.50| 0.0%, 0.52|
| 09 Hold Cup                 | 0.0%, 0.53| 0.0%, 0.39| 0.0%, 0.66| 0.0%, 0.45| 2.0%, 0.64| 0.0%, 0.46|
| 10 Collect Screws           | 0.0%, 0.26| 0.0%, 0.31| 0.0%, 0.29| 0.0%, 0.32| 0.0%, 0.26| 0.0%, 0.33|
| 11 Place Tall Box           | 0.0%, 0.61| 3.0%, 0.68| 0.0%, 0.79| 1.0%, 0.69| 6.0%, 0.83| 3.0%, 0.70|
| 12 Ball Into Bottle         | 0.0%, 0.44| 0.0%, 0.32| 0.0%, 0.49| 0.0%, 0.37| 0.0%, 0.49| 0.0%, 0.35|
| 13 Cover With Lid           | 0.0%, 0.48| 0.0%, 0.37| 0.0%, 0.53| 0.0%, 0.43| 0.0%, 0.58| 0.0%, 0.48|
| 14 Stack Cubes              | 0.0%, 0.36| 0.0%, 0.29| 0.0%, 0.40| 0.0%, 0.31| 0.0%, 0.37| 0.0%, 0.30|
| 15 Separate Marbles & Sand  |16.0%, 0.43|11.2%, 0.28|36.0%, 0.81|30.0%, 0.70|42.0%, 0.80|17.5%, 0.53|
| 16 Stand Bulb               | 0.0%, 0.54| 0.0%, 0.41| 0.0%, 0.55| 0.0%, 0.42| 0.0%, 0.55| 0.0%, 0.42|
| 17 Ball Onto Tower          | 0.0%, 0.55| 0.0%, 0.48| 0.0%, 0.62| 0.0%, 0.54| 0.0%, 0.67| 0.0%, 0.58|
| 18 Cylinder Through Hole    | 0.0%, 0.30| 0.0%, 0.26| 2.0%, 0.57| 0.8%, 0.38| 0.0%, 0.39| 0.8%, 0.41|
| 19 Stack Bowls              | 2.0%, 0.53| 0.0%, 0.30|24.0%, 0.68| 0.8%, 0.40|26.0%, 0.73| 0.8%, 0.35|
| 20 Ball Into Jar            | 6.0%, 0.55| 2.0%, 0.30| 6.0%, 0.59| 2.0%, 0.29| 6.0%, 0.55| 2.0%, 0.31|
| 21 Seal Colander            | 0.0%, 0.15| 0.0%, 0.16| 0.0%, 0.18| 0.0%, 0.18| 0.0%, 0.17| 0.0%, 0.17|
| 22 Stabilize Bottle         | 0.0%, 0.34| 0.0%, 0.36| 0.0%, 0.47| 0.0%, 0.42| 0.0%, 0.35| 0.0%, 0.38|
| 23 Place Book               | 4.0%, 0.21| 3.0%, 0.25| 2.0%, 0.30| 3.0%, 0.40| 2.0%, 0.29|26.0%, 0.54|
| 24 Raise Platform           | 4.0%, 0.29| 0.0%, 0.29|20.0%, 0.52| 0.0%, 0.32|18.0%, 0.49| 0.0%, 0.35|
| 25 Water Into Mug           | 2.0%, 0.36| 0.7%, 0.43| 0.0%, 0.36| 0.0%, 0.45| 4.0%, 0.35| 0.0%, 0.47|
| 26 Align Chopsticks         | 0.0%, 0.00| 0.0%, 0.06| 4.0%, 0.06| 0.0%, 0.08| 0.0%, 0.01| 0.0%, 0.10|
| 27 Retrieve Roll            | 0.0%, 0.20| 0.0%, 0.21| 2.0%, 0.22| 1.4%, 0.22| 0.0%, 0.21| 0.0%, 0.21|
| 28 Move Cube                |10.0%, 0.59| 0.0%, 0.45|24.0%, 0.69| 0.0%, 0.49|12.0%, 0.66| 0.7%, 0.50|
| 29 Balance Board            | 0.0%, 0.26| 0.0%, 0.34| 2.0%, 0.44| 0.0%, 0.35| 0.0%, 0.35| 0.0%, 0.38|
| 30 Differentiate Cubes      | 0.0%, 0.27| 0.0%, 0.34| 0.0%, 0.28| 0.0%, 0.40| 0.0%, 0.30| 1.0%, 0.37|
| **Average**                 | 3.8%, 0.40| 2.3%, 0.36| 8.5%, 0.50| 4.4%, 0.42| 7.8%, 0.48| 4.5%, 0.42|

## Acknowledgement

Robowits is built upon amazing open-source projects:

- [Genesis World](https://github.com/Genesis-Embodied-AI/genesis-world) Provides the universal physics engine.
- [LeRobot](https://github.com/huggingface/lerobot)  Provides training and inference infrastructures.

## Citation

If you find our work useful, please consider citing:

```
@article{lin2026robowits,
  title={RoboWits: Unexpected Challenges for Robotic Creative Problem Solving},
  author={Lin, Chunru and Zhang, Hongxin and Yu, Fenghao and Chen, Zhehuan and Griffiths, Thomas L and Choi, Yejin and Held, David and Gan, Chuang},
  journal={arXiv preprint arXiv:2605.30326},
  year={2026}
}
```