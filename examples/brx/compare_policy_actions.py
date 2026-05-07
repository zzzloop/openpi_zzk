"""Compare saved BRX policy actions against ACT HDF5 training actions."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import h5py
import numpy as np


BRX_JOINT_NAMES = [
    "FoldingModularJoint02_Joint",
    "FoldingModularJoint03_Joint",
    "Trunk_Joint",
    "ArmL02_Joint",
    "ArmL03_Joint",
    "ArmL04_Joint",
    "ArmL05_Joint",
    "ArmL06_Joint",
    "ArmL07_Joint",
    "ArmL08_Joint",
    "JawBlock01_Joint",
    "JawBlock02_Joint",
    "ArmR02_Joint",
    "ArmR03_Joint",
    "ArmR04_Joint",
    "ArmR05_Joint",
    "ArmR06_Joint",
    "ArmR07_Joint",
    "ArmR08_Joint",
    "JawBlock03_Joint",
    "JawBlock04_Joint",
    "Head02_Joint",
    "Head03_Joint",
]


def _load_train_actions(dataset_dir: Path, pattern: str) -> np.ndarray:
    paths = sorted(glob.glob(str(dataset_dir / pattern)))
    if not paths:
        raise FileNotFoundError(f"No HDF5 files matched {dataset_dir / pattern}")
    rows = []
    for path in paths:
        with h5py.File(path, "r") as episode:
            rows.append(np.asarray(episode["/action"][:], dtype=np.float32))
    actions = np.concatenate(rows, axis=0)
    if actions.ndim != 2 or actions.shape[1] != len(BRX_JOINT_NAMES):
        raise ValueError(f"Expected [T, 23] training actions, got {actions.shape}")
    print(f"[compare] loaded train actions: {actions.shape} from {len(paths)} files")
    return actions


def _load_policy_actions(path: Path) -> np.ndarray:
    actions = np.asarray(np.load(path), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != len(BRX_JOINT_NAMES):
        raise ValueError(f"Expected [T, 23] policy actions, got {actions.shape}")
    print(f"[compare] loaded policy actions: {actions.shape} from {path}")
    return actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("/home/kemove/ACT_Datasets"))
    parser.add_argument("--pattern", type=str, default="episode_*.hdf5")
    parser.add_argument("--policy-actions", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.03)
    args = parser.parse_args()

    train = _load_train_actions(args.dataset_dir, args.pattern)
    policy = _load_policy_actions(args.policy_actions)

    train_min = train.min(axis=0)
    train_max = train.max(axis=0)
    train_mean = train.mean(axis=0)
    train_std = train.std(axis=0) + 1e-6

    policy_min = policy.min(axis=0)
    policy_max = policy.max(axis=0)
    policy_mean = policy.mean(axis=0)
    policy_std = policy.std(axis=0)

    below = policy < (train_min - args.margin)
    above = policy > (train_max + args.margin)
    out_rate = (below | above).mean(axis=0)
    z_mean = np.abs((policy_mean - train_mean) / train_std)
    std_ratio = policy_std / train_std

    order = np.argsort(-(out_rate + 0.05 * z_mean))
    print("\n[compare] per-joint policy-vs-training summary")
    print("idx name train[min,max] policy[min,max] out_rate mean_z policy_std/train_std")
    for idx in order:
        print(
            f"{idx:02d} {BRX_JOINT_NAMES[idx]} "
            f"[{train_min[idx]: .4f},{train_max[idx]: .4f}] "
            f"[{policy_min[idx]: .4f},{policy_max[idx]: .4f}] "
            f"{out_rate[idx]: .3f} {z_mean[idx]: .2f} {std_ratio[idx]: .2f}"
        )

    print("\n[compare] worst joints by out-of-range rate:")
    for idx in order[:8]:
        print(f"  {idx:02d} {BRX_JOINT_NAMES[idx]} out_rate={out_rate[idx]:.3f}, mean_z={z_mean[idx]:.2f}")


if __name__ == "__main__":
    main()
