# BRX042501 pi05 微调

该路径用于 ACT 风格的 BRX HDF5 数据，数据格式如下：

- `/action`：`[T, 23]` float32，目标关节位置 qpos
- `/observations/qpos`：`[T, 23]` float32，状态关节位置 qpos
- `/observations/images/{left_eye,left_wrist,right_eye,right_wrist}`：`[T, 360, 640, 3]` uint8 RGB 图像

23 维关节顺序定义在 `openpi.policies.brx_policy.BRX_JOINT_NAMES` 中，并与 BRX Isaac Lab 控制脚本保持一致。

## 将 HDF5 转换为 LeRobot 格式

```bash
uv run examples/brx/convert_brx_hdf5_to_lerobot.py \
  --raw-dir /home/kemove/ACT_Datasets \
  --repo-id zzk/brx_act \
  --task "grab the small blocks pick them up and put them in the bucket"
```

转换后的数据集会写入：

```bash
$LEROBOT_HOME/zzk/brx_act
```

## 计算归一化统计量

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_brx_finetune
```

## 微调 pi05_base

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_brx_finetune \
  --exp-name brx_23d \
  --overwrite
```

默认配置使用 `pi05_base`，`action_dim=32`。推理后只返回前 23 维 BRX 动作。

模型侧的 padding 保持与 pi05 兼容，同时 BRX adapter 会强制输入和输出为 23 维。

## 推理冒烟测试

```bash
uv run examples/brx/test_brx_policy.py \
  --checkpoint-dir checkpoints/pi05_brx_finetune/brx_23d/29999 \
  --episode-path /home/kemove/ACT_Datasets/episode_0.hdf5 \
  --frame-index 0 \
  --prompt "grab the small blocks pick them up and put them in the bucket"
```