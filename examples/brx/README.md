# BRX042501 pi05 Fine-Tuning

This path is for ACT-style BRX HDF5 data with:

- `/action`: `[T, 23]` float32 target qpos
- `/observations/qpos`: `[T, 23]` float32 state qpos
- `/observations/images/{left_eye,left_wrist,right_eye,right_wrist}`: `[T, 360, 640, 3]` uint8 RGB

The 23D joint order is defined in `openpi.policies.brx_policy.BRX_JOINT_NAMES` and matches the BRX Isaac Lab control scripts.

## Convert HDF5 To LeRobot

```bash
uv run examples/brx/convert_brx_hdf5_to_lerobot.py \
  --raw-dir /home/kemove/ACT_Datasets \
  --repo-id zzk/brx_act \
  --task "move the object smoothly"
```

The dataset is written under `$LEROBOT_HOME/zzk/brx_act`.

## Compute Norm Stats

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_brx_finetune
```

## Fine-Tune pi05_base

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_brx_finetune \
  --exp-name brx_23d \
  --overwrite
```

The default config uses `pi05_base`, `action_dim=32`, and returns only the first 23 BRX action dimensions after inference. The model padding stays compatible with pi05 while the BRX adapter enforces 23D input/output.

## Smoke-Test Inference

```bash
uv run examples/brx/test_brx_policy.py \
  --checkpoint-dir checkpoints/pi05_brx_finetune/brx_23d/29999 \
  --episode-path /home/kemove/ACT_Datasets/episode_0.hdf5 \
  --frame-index 0 \
  --prompt "move the object smoothly"
```
