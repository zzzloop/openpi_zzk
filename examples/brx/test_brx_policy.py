"""Run one BRX policy inference call and print the 23D action chunk.

Example:
    uv run examples/brx/test_brx_policy.py \
        --checkpoint-dir checkpoints/pi05_brx_finetune/my_exp/29999 \
        --episode-path /home/kemove/ACT_Datasets/episode_0.hdf5 \
        --frame-index 0 \
        --prompt "move the object smoothly"
"""

from pathlib import Path

import h5py
import numpy as np
import tyro

from openpi.policies import policy_config
from openpi.training import config as _config


def _load_example(episode_path: Path, frame_index: int, prompt: str) -> dict:
    with h5py.File(episode_path, "r") as ep:
        return {
            "state": np.asarray(ep["/observations/qpos"][frame_index], dtype=np.float32),
            "images": {
                "left_eye": np.asarray(ep["/observations/images/left_eye"][frame_index]),
                "left_wrist": np.asarray(ep["/observations/images/left_wrist"][frame_index]),
                "right_eye": np.asarray(ep["/observations/images/right_eye"][frame_index]),
                "right_wrist": np.asarray(ep["/observations/images/right_wrist"][frame_index]),
            },
            "prompt": prompt,
        }


def main(
    checkpoint_dir: Path,
    episode_path: Path,
    *,
    config_name: str = "pi05_brx_finetune",
    frame_index: int = 0,
    prompt: str = "move the object smoothly",
) -> None:
    train_config = _config.get_config(config_name)
    policy = policy_config.create_trained_policy(train_config, str(checkpoint_dir))
    result = policy.infer(_load_example(episode_path, frame_index, prompt))
    actions = np.asarray(result["actions"], dtype=np.float32)
    print(f"inferred keys: {list(result.keys())}")
    print(f"shape of inferred action chunk: {actions.shape}")
    print(actions[:5])


if __name__ == "__main__":
    tyro.cli(main)
