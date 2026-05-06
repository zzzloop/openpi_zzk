import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import torch

from openpi.training import config as _config
from openpi.policies import policy_config

print("===== CUDA / ENV DEBUG =====")
print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"XLA_PYTHON_CLIENT_PREALLOCATE = {os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')}")
print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
print(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"torch.cuda.current_device() = {torch.cuda.current_device()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
print()

try:
    import jax
    print("===== JAX DEBUG =====")
    print(f"jax.default_backend() = {jax.default_backend()}")
    print(f"jax.devices() = {jax.devices()}")
    print(f"jax.local_devices() = {jax.local_devices()}")
    print()
except Exception as e:
    print(f"JAX import failed: {e}")
    print()

config = _config.get_config("pi05_droid")
checkpoint_dir = "/home/kemove/checkpoints/pi05_base"
policy = policy_config.create_trained_policy(config, checkpoint_dir)

print("===== POLICY DEBUG =====")
print(f"type(policy) = {type(policy)}")
print(f"policy metadata = {getattr(policy, 'metadata', None)}")
print(f"policy is_pytorch = {getattr(policy, 'is_pytorch', None)}")
print(f"policy pytorch_device = {getattr(policy, 'pytorch_device', None)}")
print()

example = {
    "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/joint_position": np.zeros(7, dtype=np.float32),
    "observation/gripper_position": np.zeros(1, dtype=np.float32),
    "prompt": "move the box smoothly"
}

policy_infer = policy.infer(example)

print(f"inferred keys: {list(policy_infer.keys())}")
print(f"policy_timing = {policy_infer.get('policy_timing')}")
action_chunk = policy_infer["actions"]
print(f"shape of inferred action chunk: {action_chunk.shape}")
print(action_chunk[:5])