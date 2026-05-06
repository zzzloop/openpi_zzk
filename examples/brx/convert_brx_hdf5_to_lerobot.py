"""Convert BRX ACT-style HDF5 episodes to LeRobot format.

Example:
    uv run examples/brx/convert_brx_hdf5_to_lerobot.py \
        --raw-dir /home/kemove/ACT_Datasets \
        --repo-id zzk/brx_act \
        --task "move the object smoothly"
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
# from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro

from openpi.policies.brx_policy import BRX_JOINT_NAMES


CAMERA_NAMES = ("left_eye", "left_wrist", "right_eye", "right_wrist")


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    fps: int = 30
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def _camera_feature(mode: Literal["video", "image"]) -> dict:
    return {
        "dtype": mode,
        "shape": (360, 640, 3),
        "names": ["height", "width", "channel"],
    }


def create_empty_dataset(
    repo_id: str,
    *,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(BRX_JOINT_NAMES),),
            "names": [list(BRX_JOINT_NAMES)],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(BRX_JOINT_NAMES),),
            "names": [list(BRX_JOINT_NAMES)],
        },
    }
    for camera in CAMERA_NAMES:
        features[f"observation.images.{camera}"] = _camera_feature(mode)

    output_path = LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=dataset_config.fps,
        robot_type="brx042501",
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def _load_images(ep: h5py.File, camera: str) -> np.ndarray:
    path = f"/observations/images/{camera}"
    if path not in ep:
        raise KeyError(f"Missing camera dataset: {path}")
    data = ep[path]
    if data.ndim == 4:
        return data[:]

    import cv2

    frames = []
    for encoded in data:
        frame = cv2.imdecode(encoded, 1)
        if frame is None:
            raise ValueError(f"Failed to decode compressed frame for {path}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return np.asarray(frames)


def load_episode(ep_path: Path) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(np.asarray(ep["/observations/qpos"][:], dtype=np.float32))
        action = torch.from_numpy(np.asarray(ep["/action"][:], dtype=np.float32))
        if state.shape[-1] != len(BRX_JOINT_NAMES):
            raise ValueError(f"{ep_path}: expected qpos dim {len(BRX_JOINT_NAMES)}, got {tuple(state.shape)}")
        if action.shape[-1] != len(BRX_JOINT_NAMES):
            raise ValueError(f"{ep_path}: expected action dim {len(BRX_JOINT_NAMES)}, got {tuple(action.shape)}")
        if state.shape[0] != action.shape[0]:
            raise ValueError(f"{ep_path}: qpos/action length mismatch {state.shape[0]} != {action.shape[0]}")
        images = {camera: _load_images(ep, camera) for camera in CAMERA_NAMES}

    for camera, frames in images.items():
        if frames.shape[0] != state.shape[0]:
            raise ValueError(f"{ep_path}: {camera} frame count {frames.shape[0]} != qpos length {state.shape[0]}")
    return images, state, action


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    *,
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    selected = range(len(hdf5_files)) if episodes is None else episodes
    for ep_idx in tqdm.tqdm(selected, desc="Converting BRX episodes"):
        ep_path = hdf5_files[ep_idx]
        images, state, action = load_episode(ep_path)
        for frame_idx in range(state.shape[0]):
            frame = {
                "observation.state": state[frame_idx],
                "action": action[frame_idx],
                "task": task,
            }
            for camera, frames in images.items():
                frame[f"observation.images.{camera}"] = frames[frame_idx]
            dataset.add_frame(frame)
        dataset.save_episode()
    return dataset


def port_brx(
    raw_dir: Path,
    repo_id: str,
    task: str = "move the object smoothly",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> None:
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw BRX dataset directory does not exist: {raw_dir}")

    hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No episode_*.hdf5 files found under {raw_dir}")
    print(f"Found {len(hdf5_files)} BRX HDF5 episodes")

    dataset = create_empty_dataset(repo_id, mode=mode, dataset_config=dataset_config)
    dataset = populate_dataset(dataset, hdf5_files, task=task, episodes=episodes)

    if push_to_hub:
        dataset.push_to_hub(
            tags=["brx042501", "act", "hdf5"],
            private=True,
            push_videos=dataset_config.use_videos,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(port_brx)
