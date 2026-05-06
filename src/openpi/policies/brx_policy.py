import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


BRX_JOINT_NAMES: tuple[str, ...] = (
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
)


def make_brx_example() -> dict:
    return {
        "state": np.zeros((len(BRX_JOINT_NAMES),), dtype=np.float32),
        "images": {
            "left_eye": np.zeros((360, 640, 3), dtype=np.uint8),
            "left_wrist": np.zeros((360, 640, 3), dtype=np.uint8),
            "right_wrist": np.zeros((360, 640, 3), dtype=np.uint8),
        },
        "prompt": "move the object smoothly",
    }


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).clip(0, 255).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected image rank 3, got shape {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image


@dataclasses.dataclass(frozen=True)
class BRXInputs(transforms.DataTransformFn):
    """Inputs for BRX 23-DoF upper-body policies.

    Expected inference input:
    - images: dict with left_eye, left_wrist, right_wrist. right_eye is accepted but unused.
    - state: 23D qpos in BRX_JOINT_NAMES order.
    - actions: optional [horizon, 23] target qpos in the same order for training.
    """

    model_type: _model.ModelType
    expected_cameras: ClassVar[tuple[str, ...]] = ("left_eye", "left_wrist", "right_wrist")

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape[-1] != len(BRX_JOINT_NAMES):
            raise ValueError(f"Expected BRX state dim {len(BRX_JOINT_NAMES)}, got {state.shape}")

        in_images = data["images"]
        missing = [name for name in self.expected_cameras if name not in in_images]
        if missing:
            raise ValueError(f"Missing BRX camera(s): {missing}")

        left_eye = _parse_image(in_images["left_eye"])
        left_wrist = _parse_image(in_images["left_wrist"])
        right_wrist = _parse_image(in_images["right_wrist"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                images = {
                    "base_0_rgb": left_eye,
                    "left_wrist_0_rgb": left_wrist,
                    "right_wrist_0_rgb": right_wrist,
                }
                image_masks = {
                    "base_0_rgb": np.True_,
                    "left_wrist_0_rgb": np.True_,
                    "right_wrist_0_rgb": np.True_,
                }
            case _:
                raise ValueError(f"Unsupported BRX model type: {self.model_type}")

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != len(BRX_JOINT_NAMES):
                raise ValueError(f"Expected BRX action dim {len(BRX_JOINT_NAMES)}, got {actions.shape}")
            inputs["actions"] = actions

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class BRXOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : len(BRX_JOINT_NAMES)], dtype=np.float32)}
