#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
lerobot-eval \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.
```
lerobot-eval \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

Note that in both examples, the repo/folder should contain at least `config.json` and `model.safetensors` files.

You can learn about the CLI options for this script in the `EvalPipelineConfig` in lerobot/configs/eval.py
"""

import concurrent.futures as cf
import json
import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from pprint import pformat
from typing import Any, TypedDict

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import Tensor, nn
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import (
    add_envs_task,
    check_env_attributes_and_types,
    close_envs,
    preprocess_observation,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, DONE, OBS_STR, REWARD
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    inside_slurm,
)


@dataclass
class DataRecorder:
    """Incrementally saves episode data to disk frame-by-frame.
    
    Records observations and info after each env.reset() and env.step(),
    saving images as individual files and other data accumulated in memory.
    After episode completion, compiles images into video and saves parquet.
    Also records final objs_info for each episode when available.
    """
    dataset_dir: Path
    fps: int
    task_desc: str = ""
    n_envs: int = 1
    env: gym.vector.VectorEnv | None = None  # Optional env reference for objs_info collection
    
    feature_names: dict = field(default_factory=dict)  # optional: key -> list of names

    # Internal state
    _episodes_info: list[dict] = field(default_factory=list)
    _image_keys: list[str] = field(default_factory=list)
    _data_keys: list[str] = field(default_factory=list)
    _feature_shapes: dict = field(default_factory=dict)  # key -> shape list
    _parquet_files: list[Path] = field(default_factory=list)
    _initialized: bool = False
    _total_frames: int = 0
    _video_threads: list[threading.Thread] = field(default_factory=list)
    
    # Per-environment episode tracking
    _env_episode_idx: list[int] = field(default_factory=list)
    _env_frame_count: list[int] = field(default_factory=list)
    _env_frame_data: list[list[dict]] = field(default_factory=list)
    _next_episode_idx: int = 0
    
    # Per-episode objs_info storage: episode_idx -> objs_info dict
    _episode_objs_info: dict = field(default_factory=dict)
    
    def __post_init__(self):
        self.data_dir = self.dataset_dir / "data"
        self.videos_dir = self.dataset_dir / "videos"
        self.frames_dir = self.dataset_dir / "frames"
        self.meta_dir = self.dataset_dir / "meta"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize per-env tracking
        self._env_episode_idx = [-1] * self.n_envs
        self._env_frame_count = [0] * self.n_envs
        self._env_frame_data = [[] for _ in range(self.n_envs)]

        # On resume: start episode index past any already-recorded episodes
        # so new episodes don't overwrite existing videos/data.
        prior_episodes_path = self.meta_dir / "episodes.json"
        if prior_episodes_path.exists():
            try:
                with open(prior_episodes_path) as f:
                    prior_episodes = json.load(f)
                if prior_episodes:
                    self._next_episode_idx = max(ep["episode_index"] for ep in prior_episodes) + 1
            except Exception:
                pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()
        return False
    
    def _init_keys(self, observation: dict):
        """Detect image vs data keys from observation and record their shapes."""
        if self._initialized:
            return
        for key, val in observation.items():
            if not isinstance(val, (np.ndarray, torch.Tensor)):
                self._data_keys.append(key)
                self._feature_shapes[key] = [1]
                continue
            shape = val.shape
            # Check if it looks like an image: 3D with valid channel dimension
            # Channel-last: (H, W, C) where C in {1, 3, 4}
            # Channel-first: (C, H, W) where C in {1, 3, 4}
            is_image = (
                "image" in key.lower() and
                len(shape) == 3 and
                (shape[-1] in (1, 3, 4) or shape[0] in (1, 3, 4))
            )
            if is_image:
                self._image_keys.append(key)
                # Normalize to HWC
                if shape[0] in (1, 3, 4) and shape[-1] not in (1, 3, 4):
                    self._feature_shapes[key] = [shape[1], shape[2], shape[0]]
                else:
                    self._feature_shapes[key] = list(shape)
            else:
                self._data_keys.append(key)
                self._feature_shapes[key] = list(shape) if len(shape) > 0 else [1]
        self._initialized = True
        print(f"[DataRecorder] Image keys: {self._image_keys}")
        print(f"[DataRecorder] Data keys: {self._data_keys}")
    
    def _save_image(self, img: np.ndarray | torch.Tensor, path: Path):
        """Save image to file, converting dtype if needed."""
        from PIL import Image
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        # Handle channel-first format (C, H, W) -> (H, W, C)
        if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
            img = np.transpose(img, (1, 2, 0))
        # Squeeze single channel dimension if present
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        Image.fromarray(img).save(path)
    
    def record_frame(
        self,
        env_idx: int,
        observation: dict,
        action: np.ndarray | None = None,
        reward: float | None = None,
        done: bool = False,
        success: bool = False,
        info: dict | None = None,
    ):
        """Record a single frame for one environment.
        
        Call after env.reset() (with action=None, reward=None) and after each env.step().
        When done=True, the episode is finalized and images compiled to video.
        
        Args:
            info: Optional info dict from env (may contain objs_info, etc.)
        """
        self._init_keys(observation)
        
        # Start new episode if needed
        if self._env_episode_idx[env_idx] == -1:
            self._env_episode_idx[env_idx] = self._next_episode_idx
            self._next_episode_idx += 1
            self._env_frame_count[env_idx] = 0
            self._env_frame_data[env_idx] = []
        
        episode_idx = self._env_episode_idx[env_idx]
        frame_idx = self._env_frame_count[env_idx]
        
        # Save images as individual files
        for img_key in self._image_keys:
            img_dir = self.frames_dir / img_key.replace(".", "_") / f"episode_{episode_idx:06d}"
            img_dir.mkdir(parents=True, exist_ok=True)
            img_path = img_dir / f"frame_{frame_idx:06d}.png"
            self._save_image(observation[img_key], img_path)
        
        # Accumulate non-image data
        frame_data = {
            "frame_index": frame_idx,
            "timestamp": frame_idx / self.fps,
        }
        for key in self._data_keys:
            val = observation[key]
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            if isinstance(val, np.ndarray):
                val = val.tolist() if val.ndim > 0 else float(val)
            frame_data[key] = val
        
        if action is not None:
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            frame_data["action"] = action.tolist() if isinstance(action, np.ndarray) else action
            if "action" not in self._feature_shapes:
                self._feature_shapes["action"] = list(np.array(action).shape)
        if reward is not None:
            frame_data["next.reward"] = float(reward)
        frame_data["next.done"] = done
        frame_data["next.success"] = success
        
        self._env_frame_data[env_idx].append(frame_data)
        self._env_frame_count[env_idx] += 1
        
        # If episode done, finalize it
        if done:
            self._finalize_episode(env_idx)
    
    def _finalize_episode(self, env_idx: int):
        """Compile episode images into video and save parquet."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        from PIL import Image
        
        episode_idx = self._env_episode_idx[env_idx]
        n_frames = self._env_frame_count[env_idx]
        frame_data_list = self._env_frame_data[env_idx]
        
        self._total_frames += n_frames
        self._episodes_info.append({
            "episode_index": episode_idx,
            "length": n_frames,
        })
        
        # Collect final objs_info if env is available
        if self.env is not None:
            try:
                inner_env = self.env.envs[env_idx] if hasattr(self.env, 'envs') else None
                if inner_env is not None and hasattr(inner_env, 'collect_objs_info'):
                    objs_info = inner_env.collect_objs_info()
                    # Convert numpy arrays to lists for JSON serialization
                    serializable_info = self._make_objs_info_serializable(objs_info)
                    self._episode_objs_info[episode_idx] = serializable_info
                    # Write incrementally so data survives interruption
                    try:
                        objs_info_path = self.data_dir / "objs_info.json"
                        existing = {}
                        if objs_info_path.exists():
                            with open(objs_info_path) as f:
                                existing = json.load(f)
                        existing[str(episode_idx)] = serializable_info
                        with open(objs_info_path, "w") as f:
                            json.dump(existing, f, indent=2)
                    except Exception as e:
                        logging.debug(f"Could not write objs_info.json after episode {episode_idx}: {e}")
            except Exception as e:
                logging.debug(f"Could not collect objs_info for episode {episode_idx}: {e}")
        
        # Compile images into video (in background threads)
        for img_key in self._image_keys:
            frames_episode_dir = self.frames_dir / img_key.replace(".", "_") / f"episode_{episode_idx:06d}"
            video_dir = self.videos_dir / img_key.replace(".", "_")
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"episode_{episode_idx:06d}.mp4"
            
            # Load all frames
            frames = []
            for i in range(n_frames):
                img_path = frames_episode_dir / f"frame_{i:06d}.png"
                frames.append(np.array(Image.open(img_path)))
            frames_array = np.stack(frames)
            
            # Write video and cleanup frames in background thread
            thread = threading.Thread(
                target=self._write_video_and_cleanup,
                args=(str(video_path), frames_array, self.fps, frames_episode_dir),
            )
            thread.start()
            self._video_threads.append(thread)
        
        # Save parquet for this episode
        def _to_serializable(v):
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            if isinstance(v, np.ndarray):
                return v.flatten().tolist()
            return v

        parquet_data = {}
        keys = list(frame_data_list[0].keys())
        for key in keys:
            parquet_data[key] = [_to_serializable(f[key]) for f in frame_data_list]
        
        # Add episode_index and global index
        parquet_data["episode_index"] = [episode_idx] * n_frames
        start_idx = self._total_frames - n_frames
        parquet_data["index"] = list(range(start_idx, start_idx + n_frames))
        
        # Add video paths as references
        for img_key in self._image_keys:
            video_path_str = f"videos/{img_key.replace('.', '_')}/episode_{episode_idx:06d}.mp4"
            parquet_data[f"{img_key}_path"] = [video_path_str] * n_frames
        
        table = pa.table(parquet_data)
        parquet_path = self.data_dir / f"episode_{episode_idx:06d}.parquet"
        pq.write_table(table, parquet_path)
        self._parquet_files.append(parquet_path)
        
        # Reset env state for next episode
        self._env_episode_idx[env_idx] = -1
        self._env_frame_count[env_idx] = 0
        self._env_frame_data[env_idx] = []

        # Write info.json incrementally so it's valid even if eval crashes mid-run
        self._write_info_json(self._episodes_info)
    
    def _write_info_json(self, all_episodes: list):
        """Write v3.0-compatible info.json for the episodes seen so far."""
        features = {}
        for key in self._data_keys:
            features[key] = {
                "dtype": "float32",
                "shape": self._feature_shapes.get(key, [1]),
                "names": self.feature_names.get(key),
                "video_info": None,
            }
        if "action" in self._feature_shapes:
            features["action"] = {
                "dtype": "float32",
                "shape": self._feature_shapes["action"],
                "names": self.feature_names.get("action"),
                "video_info": None,
            }
        for key in self._image_keys:
            features[key] = {
                "names": ["height", "width", "channels"],
                "shape": self._feature_shapes.get(key, [480, 848, 3]),
                "dtype": "video",
                "video_info": {
                    "video.fps": float(self.fps),
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }
        total_episodes = len(all_episodes)
        meta_info = {
            "codebase_version": "v3.0",
            "robot_type": None,
            "total_episodes": total_episodes,
            "total_frames": sum(ep.get("length", 0) for ep in all_episodes),
            "total_tasks": 1,
            "fps": self.fps,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": "data/train-00000-of-00001.parquet",
            "features": features,
        }
        with open(self.meta_dir / "info.json", "w") as f:
            json.dump(meta_info, f, indent=2)

    def _write_video_and_cleanup(self, video_path: str, frames: np.ndarray, fps: int, frames_dir: Path):
        """Write video and clean up individual frame files."""
        write_video(video_path, frames, fps)
        # Clean up frame files
        import shutil
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
    
    def _make_objs_info_serializable(self, objs_info: dict) -> dict:
        """Convert objs_info dict to JSON-serializable format."""
        result = {}
        for obj_name, obj_data in objs_info.items():
            result[obj_name] = {}
            for key, value in obj_data.items():
                if isinstance(value, np.ndarray):
                    result[obj_name][key] = value.tolist()
                elif isinstance(value, torch.Tensor):
                    result[obj_name][key] = value.cpu().numpy().tolist()
                else:
                    result[obj_name][key] = value
        return result
    
    def discard_episode(self):
        """Discard any partially-recorded episode state without saving."""
        for env_idx in range(self.n_envs):
            if self._env_episode_idx[env_idx] != -1:
                # Do NOT decrement _next_episode_idx: the slot is abandoned so
                # parquet episode indices stay in sync with batch_ix in eval_info.
                self._env_episode_idx[env_idx] = -1
                self._env_frame_count[env_idx] = 0
                self._env_frame_data[env_idx] = []

    def finalize(self):
        """Wait for all video threads and write final metadata."""
        import pyarrow.parquet as pq
        
        # Finalize any incomplete episodes (shouldn't happen normally)
        for env_idx in range(self.n_envs):
            if self._env_episode_idx[env_idx] != -1 and self._env_frame_count[env_idx] > 0:
                logging.warning(f"Finalizing incomplete episode for env {env_idx}")
                self._finalize_episode(env_idx)
        
        # Wait for all video saving threads
        for thread in self._video_threads:
            thread.join()
        
        # Clean up frames directory if empty
        if self.frames_dir.exists():
            import shutil
            try:
                shutil.rmtree(self.frames_dir)
            except OSError:
                pass  # Directory not empty, keep it
        
        # Merge parquet files into single file (for compatibility).
        # If resuming, prepend the existing merged file (or any unmerged per-episode
        # parquets from a prior killed run) so prior episodes are preserved.
        if self._parquet_files:
            import pyarrow as pa
            merged_path = self.data_dir / "train-00000-of-00001.parquet"
            written_set = set(self._parquet_files)
            unmerged_prior = sorted(
                p for p in self.data_dir.glob("episode_*.parquet") if p not in written_set
            )
            prior = []
            if merged_path.exists():
                prior = [pq.read_table(merged_path)]
            elif unmerged_prior:
                prior = [pq.read_table(p) for p in unmerged_prior]
            tables = prior + [pq.read_table(f) for f in self._parquet_files]
            pq.write_table(pa.concat_tables(tables), merged_path)
            for f in self._parquet_files:
                f.unlink()
            for f in unmerged_prior:
                f.unlink()
        
        # Merge episodes metadata with any existing episodes from a prior run.
        prior_episodes_path = self.meta_dir / "episodes.json"
        if prior_episodes_path.exists():
            try:
                with open(prior_episodes_path) as f:
                    prior_episodes = json.load(f)
            except Exception:
                prior_episodes = []
        else:
            prior_episodes = []
        all_episodes = prior_episodes + self._episodes_info

        self._write_info_json(all_episodes)
        with open(prior_episodes_path, "w") as f:
            json.dump(all_episodes, f, indent=2)
        
        # Write objs_info.json with final object states for each episode
        if self._episode_objs_info:
            objs_info_path = self.data_dir / "objs_info.json"
            # Merge with any existing objs_info from prior runs
            existing_objs_info = {}
            if objs_info_path.exists():
                try:
                    with open(objs_info_path) as f:
                        existing_objs_info = json.load(f)
                except Exception:
                    pass
            # Merge: new episodes override existing ones with same index
            existing_objs_info.update({str(k): v for k, v in self._episode_objs_info.items()})
            with open(objs_info_path, "w") as f:
                json.dump(existing_objs_info, f, indent=2)
            logging.info(f"Saved objs_info for {len(self._episode_objs_info)} episodes to {objs_info_path}")
        
        logging.info(f"Finalized incremental dataset at {self.dataset_dir}")


def _index_obs_value(v, idx):
    """Index into an observation value by env index, handling nested dicts."""
    if isinstance(v, dict):
        return {k: _index_obs_value(sub_v, idx) for k, sub_v in v.items()}
    return v[idx]


def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
    data_recorder: DataRecorder | None = None,
) -> dict:
    """Run a batched policy rollout once through a batch of environments.

    Note that all environments in the batch are run until the last environment is done. This means some
    data will probably need to be discarded (for environments that aren't the first one to be done).

    The return dictionary contains:
        (optional) "observation": A dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.

    Args:
        env: The batch of environments.
        policy: The policy. Must be a PyTorch nn module.
        seeds: The environments are seeded once at the start of the rollout. If provided, this argument
            specifies the seeds for each of the environments.
        return_observations: Whether to include all observations in the returned rollout data. Observations
            are returned optionally because they typically take more memory to cache. Defaults to False.
        render_callback: Optional rendering callback to be used after the environments are reset, and after
            every step.
        data_recorder: Optional DataRecorder for frame-by-frame data recording.
    Returns:
        The dictionary described above.
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    # Reset the policy and environments.
    policy.reset()
    observation_raw, info = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env)
    
    # Preprocess initial observation once - used for both recording and policy
    observation = preprocess_observation(observation_raw)
    
    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    step = 0
    # Keep track of which environments are done.
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),  # we dont want progress bar when we use slurm, since it clutters the logs
        leave=False,
    )
    check_env_attributes_and_types(env)
    
    while not np.all(done) and step < max_steps:
        # observation is already preprocessed (from reset or previous step)
        if return_observations:
            all_observations.append(deepcopy(observation))

        # Save obs_t before policy transforms, to pair with action_t in recording
        pre_step_observation = observation

        # Infer "task" from attributes of environments.
        # TODO: works with SyncVectorEnv but not AsyncVectorEnv
        observation = add_envs_task(env, observation)
        pre_step_observation["task"] = observation["task"]

        # Apply environment-specific preprocessing (e.g., LiberoProcessorStep for LIBERO)
        observation = env_preprocessor(observation)

        observation = preprocessor(observation)
        # Move observation tensors to policy device
        policy_device = next(policy.parameters()).device
        observation = {k: v.to(policy_device) if isinstance(v, torch.Tensor) else v for k, v in observation.items()}
        with torch.inference_mode():
            action = policy.select_action(observation)
        action = postprocessor(action)

        action_transition = {ACTION: action}
        action_transition = env_postprocessor(action_transition)
        action = action_transition[ACTION]

        # Convert to CPU / numpy.
        action_numpy: np.ndarray = action.to("cpu").numpy()
        assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"

        # Apply the next action.
        observation_raw, reward, terminated, truncated, info = env.step(action_numpy)
        if render_callback is not None:
            render_callback(env)

        # VectorEnv stores is_success in `info["final_info"][env_index]["is_success"]`. "final_info" isn't
        # available if none of the envs finished, or when using AutoresetMode.DISABLED.
        if "final_info" in info:
            final_info = info["final_info"]
            if not isinstance(final_info, dict):
                raise RuntimeError(
                    "Unsupported `final_info` format: expected dict (Gymnasium >= 1.0). "
                    "You're likely using an older version of gymnasium (< 1.0). Please upgrade."
                )
            successes = final_info["is_success"].tolist()
        elif "is_success" in info:
            # AutoresetMode.DISABLED: is_success is in the step info directly on termination
            is_success = info["is_success"]
            if isinstance(is_success, (np.ndarray, torch.Tensor)):
                successes = [bool(v) for v in is_success]
            elif isinstance(is_success, list):
                successes = [bool(v) for v in is_success]
            else:
                successes = [bool(is_success)] * env.num_envs
        else:
            successes = [False] * env.num_envs

        # Keep track of which environments are done so far.
        # Mark the episode as done if we reach the maximum step limit.
        # This ensures that the rollout always terminates cleanly at `max_steps`,
        # and allows logging/saving (e.g., videos) to be triggered consistently.
        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)

        # Preprocess observation once - used for both recording and next iteration's policy
        observation = preprocess_observation(observation_raw)

        # Record obs_t + action_t (pre-step obs paired with the action taken from it)
        if data_recorder is not None:
            for env_idx in range(env.num_envs):
                env_obs = {k: _index_obs_value(v, env_idx) for k, v in pre_step_observation.items()}
                # Extract per-env info from VectorEnv info dict (only index array-like values)
                env_info = {k: v[env_idx] if isinstance(v, (list, np.ndarray, torch.Tensor)) else v 
                            for k, v in info.items()} if info else None
                data_recorder.record_frame(
                    env_idx,
                    env_obs,
                    action_numpy[env_idx],
                    float(reward[env_idx]),
                    bool(done[env_idx]),
                    successes[env_idx],
                    env_info,
                )

        all_actions.append(torch.from_numpy(action_numpy))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes, device='cpu'))

        step += 1
        running_success_rate = (
            einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").cpu().numpy().mean()
        )
        progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
        progbar.update()

    # Track the final observation (already preprocessed from last step).
    if return_observations:
        all_observations.append(deepcopy(observation))

    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
    ret = {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
    if return_observations:
        stacked_observations = {}
        for key in all_observations[0]:
            stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret[OBS_STR] = stacked_observations

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    return ret


def _write_partial_eval_info(
    output_dir: Path,
    sum_rewards: list,
    max_rewards: list,
    all_successes: list,
    all_seeds: list,
    n_episodes: int,
    start: float,
) -> None:
    # Filter out None values (episodes not yet evaluated) and build per_episode list
    per_episode = []
    valid_sum_rewards = []
    valid_max_rewards = []
    valid_successes = []
    for i in range(min(len(sum_rewards), n_episodes)):
        if sum_rewards[i] is not None:
            per_episode.append({
                "episode_ix": i,
                "sum_reward": sum_rewards[i],
                "max_reward": max_rewards[i],
                "success": all_successes[i],
                "seed": all_seeds[i],
            })
            valid_sum_rewards.append(sum_rewards[i])
            valid_max_rewards.append(max_rewards[i])
            valid_successes.append(all_successes[i])

    n_done = len(per_episode)
    info = {
        "per_episode": per_episode,
        "overall": {
            "avg_sum_reward": float(np.nanmean(valid_sum_rewards)) if n_done else float("nan"),
            "avg_max_reward": float(np.nanmean(valid_max_rewards)) if n_done else float("nan"),
            "pc_success": float(np.nanmean(valid_successes) * 100) if n_done else float("nan"),
            "n_episodes": n_done,
            "n_episodes_total": n_episodes,
            "eval_s": time.time() - start,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eval_task_info.json", "w") as f:
        json.dump(info, f, indent=2)


def eval_policy(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    data_recorder: DataRecorder | None = None,
    output_dir: Path | None = None,
) -> dict:
    """
    Args:
        env: The batch of environments.
        policy: The policy.
        n_episodes: The number of episodes to evaluate.
        max_episodes_rendered: Maximum number of episodes to render into videos.
        videos_dir: Where to save rendered videos.
        return_episode_data: Whether to return episode data for online training. Incorporates the data into
            the "episodes" key of the returned dictionary.
        start_seed: The first seed to use for the first individual rollout. For all subsequent rollouts the
            seed is incremented by 1. If not provided, the environments are not manually seeded.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    if not isinstance(policy, PreTrainedPolicy):
        exc = ValueError(
            f"Policy of type 'PreTrainedPolicy' is expected, but type '{type(policy)}' was provided."
        )
        try:
            from peft import PeftModel

            if not isinstance(policy, PeftModel):
                raise exc
        except ImportError:
            raise exc from None

    start = time.time()
    policy.eval()

    # Determine how many batched rollouts we need to get n_episodes. Note that if n_episodes is not evenly
    # divisible by env.num_envs we end up discarding some data in the last batch.
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    # Resume from existing partial results if output_dir already has eval_task_info.json.
    # Now supports skipping any existing episode (not just sequential from start).
    sum_rewards: list[float | None] = [None] * n_episodes
    max_rewards: list[float | None] = [None] * n_episodes
    all_successes: list[bool | None] = [None] * n_episodes
    all_seeds: list[int | None] = [None] * n_episodes
    existing_episodes: set[int] = set()
    if output_dir is not None:
        partial_path = output_dir / "eval_task_info.json"
        if partial_path.exists():
            try:
                with open(partial_path) as f:
                    partial = json.load(f)
                for ep in partial.get("per_episode", []):
                    ep_ix = ep.get("episode_ix", -1)
                    if 0 <= ep_ix < n_episodes:
                        sum_rewards[ep_ix] = ep["sum_reward"]
                        max_rewards[ep_ix] = ep["max_reward"]
                        all_successes[ep_ix] = ep["success"]
                        all_seeds[ep_ix] = ep.get("seed")
                        existing_episodes.add(ep_ix)
                if existing_episodes:
                    logging.info(f"Resuming eval: found {len(existing_episodes)} existing episodes, will skip them.")
            except Exception as e:
                logging.warning(f"Could not load existing eval_task_info.json for resuming: {e}")

    threads = []  # for video saving threads
    n_episodes_rendered = 0  # for saving the correct number of videos

    # Callback for visualization.
    def render_frame(env: gym.vector.VectorEnv):
        # noqa: B023
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if isinstance(env, gym.vector.SyncVectorEnv):
            ep_frames.append(np.stack([env.envs[i].render() for i in range(n_to_render_now)]))  # noqa: B023
        elif isinstance(env, gym.vector.AsyncVectorEnv):
            # Here we must render all frames and discard any we don't need.
            ep_frames.append(np.stack(env.call("render")[:n_to_render_now]))

    if max_episodes_rendered > 0:
        video_paths: list[str] = []

    if return_episode_data:
        episode_data: dict | None = None

    # Track data index for episode compilation (needed for both in-memory and incremental saving)
    next_data_index = 0

    # we dont want progress bar when we use slurm, since it clutters the logs
    progbar = trange(0, n_batches, desc="Stepping through eval batches", disable=inside_slurm())
    for batch_ix in progbar:
        # Skip episodes that already have results (enables re-running only missing/deleted episodes)
        if batch_ix in existing_episodes:
            continue
        # Cache frames for rendering videos. Each item will be (b, h, w, c), and the list indexes the rollout
        # step.
        if max_episodes_rendered > 0:
            ep_frames: list[np.ndarray] = []

        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)
            )
        try:
            rollout_data = rollout(
                env=env,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                seeds=list(seeds) if seeds else None,
                return_observations=return_episode_data and data_recorder is None,
                render_callback=render_frame if max_episodes_rendered > 0 else None,
                data_recorder=data_recorder,
            )
        except Exception as e:
            logging.warning(f"Episode {batch_ix} failed with {type(e).__name__}: {e}. Treating as failed episode.")
            if data_recorder is not None:
                data_recorder.discard_episode()
            # Store failed episode results at the correct index
            for i in range(env.num_envs):
                ep_ix = batch_ix * env.num_envs + i
                if ep_ix < n_episodes:
                    sum_rewards[ep_ix] = 0.0
                    max_rewards[ep_ix] = 0.0
                    all_successes[ep_ix] = False
                    all_seeds[ep_ix] = list(seeds)[i] if seeds else None
            if output_dir is not None:
                _write_partial_eval_info(output_dir, sum_rewards, max_rewards, all_successes, all_seeds, n_episodes, start)
            continue

        # Figure out where in each rollout sequence the first done condition was encountered (results after
        # this won't be included).
        n_steps = rollout_data["done"].shape[1]
        # Note: this relies on a property of argmax: that it returns the first occurrence as a tiebreaker.
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

        # Make a mask with shape (batch, n_steps) to mask out rollout data after the first done
        # (batch-element-wise). Note the `done_indices + 1` to make sure to keep the data from the done step.
        mask = (torch.arange(n_steps, device='cpu') <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
        # Compute metrics for this batch.
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum").tolist()
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max").tolist()
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any").tolist()
        # Store results at the correct indices (supports skipping existing episodes)
        for i in range(env.num_envs):
            ep_ix = batch_ix * env.num_envs + i
            if ep_ix < n_episodes:
                sum_rewards[ep_ix] = batch_sum_rewards[i]
                max_rewards[ep_ix] = batch_max_rewards[i]
                all_successes[ep_ix] = batch_successes[i]
                all_seeds[ep_ix] = list(seeds)[i] if seeds else None

        if output_dir is not None:
            _write_partial_eval_info(output_dir, sum_rewards, max_rewards, all_successes, all_seeds, n_episodes, start)

        # Compile episode data for in-memory return (legacy behavior, only when no data_recorder)
        if return_episode_data and data_recorder is None:
            this_batch_data = _compile_episode_data(
                rollout_data,
                done_indices,
                start_episode_index=batch_ix * env.num_envs,
                start_data_index=next_data_index,
                fps=env.unwrapped.metadata["render_fps"],
            )
            next_data_index = this_batch_data["index"][-1].item() + 1
            # Legacy behavior: accumulate in memory
            if episode_data is None:
                episode_data = this_batch_data
            else:
                assert episode_data["episode_index"][-1] + 1 == this_batch_data["episode_index"][0]
                assert episode_data["index"][-1] + 1 == this_batch_data["index"][0]
                episode_data = {k: torch.cat([episode_data[k], this_batch_data[k]]) for k in episode_data}

        # Maybe render video for visualization.
        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            batch_stacked_frames = np.stack(ep_frames, axis=1)  # (b, t, *)
            for stacked_frames, done_index in zip(
                batch_stacked_frames, done_indices.flatten().tolist(), strict=False
            ):
                if n_episodes_rendered >= max_episodes_rendered:
                    break

                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        stacked_frames[: done_index + 1],  # + 1 to capture the last observation
                        env.unwrapped.metadata["render_fps"],
                    ),
                )
                thread.start()
                threads.append(thread)
                n_episodes_rendered += 1

        done_successes = [s for s in all_successes[:n_episodes] if s is not None]
        running_sr = np.mean(done_successes) * 100 if done_successes else 0.0
        progbar.set_postfix({"running_success_rate": f"{running_sr:.1f}%"})

    # Wait till all video rendering threads are done.
    for thread in threads:
        thread.join()

    # Compile eval info.
    info = {
        "per_episode": [
            {
                "episode_ix": i,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": success,
                "seed": seed,
            }
            for i, (sum_reward, max_reward, success, seed) in enumerate(
                zip(
                    sum_rewards[:n_episodes],
                    max_rewards[:n_episodes],
                    all_successes[:n_episodes],
                    all_seeds[:n_episodes],
                    strict=True,
                )
            )
        ],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
        },
    }

    if return_episode_data:
        info["episodes"] = episode_data

    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths

    return info


def _compile_episode_data(
    rollout_data: dict, done_indices: Tensor, start_episode_index: int, start_data_index: int, fps: float
) -> dict:
    """Convenience function for `eval_policy(return_episode_data=True)`

    Compiles all the rollout data into a Hugging Face dataset.

    Similar logic is implemented when datasets are pushed to hub (see: `push_to_hub`).
    """
    ep_dicts = []
    total_frames = 0
    for ep_ix in range(rollout_data[ACTION].shape[0]):
        # + 2 to include the first done frame and the last observation frame.
        num_frames = done_indices[ep_ix].item() + 2
        total_frames += num_frames

        # Here we do `num_frames - 1` as we don't want to include the last observation frame just yet.
        ep_dict = {
            ACTION: rollout_data[ACTION][ep_ix, : num_frames - 1],
            "episode_index": torch.tensor([start_episode_index + ep_ix] * (num_frames - 1)),
            "frame_index": torch.arange(0, num_frames - 1, 1),
            "timestamp": torch.arange(0, num_frames - 1, 1) / fps,
            DONE: rollout_data["done"][ep_ix, : num_frames - 1],
            "next.success": rollout_data["success"][ep_ix, : num_frames - 1],
            REWARD: rollout_data["reward"][ep_ix, : num_frames - 1].type(torch.float32),
        }

        # For the last observation frame, all other keys will just be copy padded.
        for k in ep_dict:
            ep_dict[k] = torch.cat([ep_dict[k], ep_dict[k][-1:]])

        for key in rollout_data[OBS_STR]:
            ep_dict[key] = rollout_data[OBS_STR][key][ep_ix, :num_frames]

        ep_dicts.append(ep_dict)

    data_dict = {}
    for key in ep_dicts[0]:
        data_dict[key] = torch.cat([x[key] for x in ep_dicts])

    data_dict["index"] = torch.arange(start_data_index, start_data_index + total_frames, 1)

    return data_dict

@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Making environment.")
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )

    logging.info("Making policy.")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=cfg.rename_map,
    )

    policy.eval()

    # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    # Create environment-specific preprocessor and postprocessor (e.g., for LIBERO environments)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    # Set up base output directory for recording when enabled
    recording_dir = Path(cfg.output_dir) if cfg.eval.recording else None
    task_desc = getattr(cfg.env, "task", None) or ""

    # Load feature names from training dataset info.json for consistent recording metadata
    feature_names = {}
    if recording_dir is not None:
        try:
            dataset_path = getattr(cfg.dataset, "root", None) or getattr(cfg.dataset, "repo_id", None)
            if dataset_path:
                info_path = Path(dataset_path) / "meta" / "info.json"
                if info_path.exists():
                    with open(info_path) as _f:
                        _info = json.load(_f)
                    feature_names = {k: v.get("names") for k, v in _info.get("features", {}).items() if v.get("names")}
        except Exception:
            pass

    # When recording is enabled, disable separate video rendering to avoid duplicates
    if recording_dir:
        max_episodes_rendered = 0
        videos_dir = None
    else:
        max_episodes_rendered = 10
        videos_dir = Path(cfg.output_dir) / "videos"

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=cfg.eval.n_episodes,
            max_episodes_rendered=max_episodes_rendered,
            videos_dir=videos_dir,
            return_episode_data=cfg.eval.return_episode_data,
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
            recording_dir=recording_dir,
            task_desc=task_desc,
            output_dir=Path(cfg.output_dir),
            feature_names=feature_names,
        )
        print("Overall Aggregated Metrics:")
        print(info["overall"])

        # Print per-suite stats
        for task_group, task_group_info in info.items():
            print(f"\nAggregated Metrics for {task_group}:")
            print(task_group_info)
    # Close all vec envs
    close_envs(envs)

    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    logging.info("End of eval")


# ---- typed payload returned by one task eval ----
class TaskMetrics(TypedDict, total=False):
    sum_rewards: list[float]
    max_rewards: list[float]
    successes: list[bool]
    video_paths: list[str]
    episodes: dict  # Episode data with observations, actions, etc.


ACC_KEYS = ("sum_rewards", "max_rewards", "successes", "video_paths")


def eval_one(
    env: gym.vector.VectorEnv,
    *,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
    data_recorder: DataRecorder | None = None,
    output_dir: Path | None = None,
) -> TaskMetrics:
    """Evaluates one task_id of one suite using the provided vec env."""

    task_videos_dir = videos_dir

    task_result = eval_policy(
        env=env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
        data_recorder=data_recorder,
        output_dir=output_dir,
    )

    per_episode = task_result["per_episode"]
    result = TaskMetrics(
        sum_rewards=[ep["sum_reward"] for ep in per_episode],
        max_rewards=[ep["max_reward"] for ep in per_episode],
        successes=[ep["success"] for ep in per_episode],
        video_paths=task_result.get("video_paths", []),
    )
    if "episodes" in task_result:
        result["episodes"] = task_result["episodes"]
    return result


def run_one(
    task_group: str,
    task_id: int,
    env,
    *,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
    recording_dir: Path | None = None,
    task_desc: str = "",
    output_dir: Path | None = None,
    feature_names: dict | None = None,
):
    """
    Run eval_one for a single (task_group, task_id, env).
    Returns (task_group, task_id, task_metrics_dict).
    """
    task_videos_dir = None
    if videos_dir is not None:
        task_videos_dir = videos_dir / f"{task_group}_{task_id}"
        task_videos_dir.mkdir(parents=True, exist_ok=True)

    # Create DataRecorder context if recording_dir is provided, otherwise use nullcontext
    if recording_dir is not None:
        task_recording_dir = recording_dir / f"{task_group}_{task_id}" / "recordings"
        fps = env.unwrapped.metadata.get("render_fps", 30)
        recorder_ctx = DataRecorder(
            dataset_dir=task_recording_dir,
            fps=fps,
            task_desc=task_desc or f"{task_group}_{task_id}",
            n_envs=env.num_envs,
            env=env,  # Pass env for objs_info collection
            feature_names=feature_names or {},
        )
    else:
        recorder_ctx = nullcontext()

    with recorder_ctx as data_recorder:
        metrics = eval_one(
            env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=n_episodes,
            max_episodes_rendered=max_episodes_rendered,
            videos_dir=task_videos_dir,
            return_episode_data=return_episode_data,
            start_seed=start_seed,
            data_recorder=data_recorder,
            output_dir=output_dir,
        )

    if max_episodes_rendered > 0:
        metrics.setdefault("video_paths", [])
    return task_group, task_id, metrics


def eval_policy_all(
    envs: dict[str, dict[int, gym.vector.VectorEnv]],
    policy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    *,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    max_parallel_tasks: int = 1,
    recording_dir: Path | None = None,
    task_desc: str = "",
    output_dir: Path | None = None,
    feature_names: dict | None = None,
) -> dict:
    """
    Evaluate a nested `envs` dict: {task_group: {task_id: vec_env}}.
    This implementation flattens tasks, runs them sequentially or via ThreadPoolExecutor,
    accumulates per-group and overall statistics, and returns the same aggregate metrics
    schema as the single-env evaluator (avg_sum_reward / avg_max_reward / pc_success / timings)
    plus per-task infos.
    """
    start_t = time.time()

    # Flatten envs into list of (task_group, task_id, env)
    tasks = [(tg, tid, vec) for tg, group in envs.items() for tid, vec in group.items()]

    # accumulators: track metrics at both per-group level and across all groups
    group_acc: dict[str, dict[str, list]] = defaultdict(lambda: {k: [] for k in ACC_KEYS})
    overall: dict[str, list] = {k: [] for k in ACC_KEYS}
    per_task_infos: list[dict] = []

    # small inline helper to accumulate one task's metrics into accumulators
    def _accumulate_to(group: str, metrics: dict):
        # metrics expected to contain 'sum_rewards', 'max_rewards', 'successes', optionally 'video_paths'
        # but eval_one may store per-episode lists; we assume metrics uses scalars averaged per task as before.
        # To be robust, accept scalars or lists.
        def _append(key, value):
            if value is None:
                return
            if isinstance(value, list):
                group_acc[group][key].extend(value)
                overall[key].extend(value)
            else:
                group_acc[group][key].append(value)
                overall[key].append(value)

        _append("sum_rewards", metrics.get("sum_rewards"))
        _append("max_rewards", metrics.get("max_rewards"))
        _append("successes", metrics.get("successes"))
        # video_paths is list-like
        paths = metrics.get("video_paths", [])
        if paths:
            group_acc[group]["video_paths"].extend(paths)
            overall["video_paths"].extend(paths)

    # Helper to compute aggregated metrics
    def _agg_from_list(xs):
        if not xs:
            return float("nan")
        arr = np.array(xs, dtype=float)
        return float(np.nanmean(arr))

    # Helper to save current eval info incrementally
    def _save_eval_info():
        if output_dir is None:
            return
        # Compute current per-group aggregates
        groups_aggregated = {}
        for group, acc in group_acc.items():
            groups_aggregated[group] = {
                "avg_sum_reward": _agg_from_list(acc["sum_rewards"]),
                "avg_max_reward": _agg_from_list(acc["max_rewards"]),
                "pc_success": _agg_from_list(acc["successes"]) * 100 if acc["successes"] else float("nan"),
                "n_episodes": len(acc["sum_rewards"]),
                "video_paths": list(acc["video_paths"]),
            }
        # Current overall aggregates
        overall_agg = {
            "avg_sum_reward": _agg_from_list(overall["sum_rewards"]),
            "avg_max_reward": _agg_from_list(overall["max_rewards"]),
            "pc_success": _agg_from_list(overall["successes"]) * 100 if overall["successes"] else float("nan"),
            "n_episodes": len(overall["sum_rewards"]),
            "eval_s": time.time() - start_t,
            "eval_ep_s": (time.time() - start_t) / max(1, len(overall["sum_rewards"])),
            "video_paths": list(overall["video_paths"]),
        }
        info = {
            "overall": overall_agg,
            "per_group": groups_aggregated,
            "per_task": per_task_infos,
        }
        with open(output_dir / "eval_info.json", "w") as f:
            json.dump(info, f, indent=2)

    # Always pass output_dir so eval_policy() can write per-episode eval_task_info.json (supports resume).
    # Each task gets its own subdirectory under output_dir (e.g., robowits_0/).
    single_task = len(tasks) == 1

    def _get_task_output_dir(task_group: str, task_id: int) -> Path | None:
        if output_dir is None:
            return None
        return output_dir / f"{task_group}_{task_id}"

    task_runner = partial(
        run_one,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
        recording_dir=recording_dir,
        task_desc=task_desc,
        feature_names=feature_names,
        # output_dir is passed dynamically per task below
    )

    if max_parallel_tasks <= 1:
        # sequential path (single accumulator path on the main thread)
        # NOTE: keeping a single-threaded accumulator avoids concurrent list appends or locks
        for task_group, task_id, env in tasks:
            tg, tid, metrics = task_runner(task_group, task_id, env, output_dir=_get_task_output_dir(task_group, task_id))
            _accumulate_to(tg, metrics)
            per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})
            _save_eval_info()
    else:
        # threaded path: submit all tasks, consume completions on main thread and accumulate there
        with cf.ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
            fut2meta = {}
            for task_group, task_id, env in tasks:
                fut = executor.submit(task_runner, task_group, task_id, env, output_dir=_get_task_output_dir(task_group, task_id))
                fut2meta[fut] = (task_group, task_id)
            for fut in cf.as_completed(fut2meta):
                tg, tid, metrics = fut.result()
                _accumulate_to(tg, metrics)
                per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})
                _save_eval_info()

    # compute per-group aggregates
    groups_aggregated = {}
    for group, acc in group_acc.items():
        groups_aggregated[group] = {
            "avg_sum_reward": _agg_from_list(acc["sum_rewards"]),
            "avg_max_reward": _agg_from_list(acc["max_rewards"]),
            "pc_success": _agg_from_list(acc["successes"]) * 100 if acc["successes"] else float("nan"),
            "n_episodes": len(acc["sum_rewards"]),
            "video_paths": list(acc["video_paths"]),
        }

    # overall aggregates
    overall_agg = {
        "avg_sum_reward": _agg_from_list(overall["sum_rewards"]),
        "avg_max_reward": _agg_from_list(overall["max_rewards"]),
        "pc_success": _agg_from_list(overall["successes"]) * 100 if overall["successes"] else float("nan"),
        "n_episodes": len(overall["sum_rewards"]),
        "eval_s": time.time() - start_t,
        "eval_ep_s": (time.time() - start_t) / max(1, len(overall["sum_rewards"])),
        "video_paths": list(overall["video_paths"]),
    }

    return {
        "overall": overall_agg,
        "per_group": groups_aggregated,
        "per_task": per_task_infos,
    }


def main():
    init_logging()
    register_third_party_plugins()
    eval_main()


if __name__ == "__main__":
    main()
