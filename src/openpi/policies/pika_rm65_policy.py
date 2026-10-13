import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_pika_rm65_example() -> dict:
    """Create an inference observation for the reconstructed Pika RM65 policy."""
    return {
        "state": np.zeros(7, dtype=np.float32),
        "center_image": np.zeros((360, 640, 3), dtype=np.uint8),
        "left_image": np.zeros((360, 640, 3), dtype=np.uint8),
        "prompt": "pick and place",
    }


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class PikaRM65Inputs(transforms.DataTransformFn):
    """Map Pika RM65 single-arm observations to the pi0.5 model schema.

    Dataset cameras are `observation.images.center` and `observation.images.left`.
    Inference should send 7-D joint+gripper state in degrees, matching the checkpoint
    `norm_stats` (`arm_a_joint_1..6` + `arm_a_gripper`).
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        raw_state = np.asarray(data["state"])
        if raw_state.shape[-1] != 7:
            raise ValueError(f"Expected 7-dimensional Pika RM65 state, got {raw_state.shape}")

        base_image = _parse_image(data["center_image"])
        wrist_image = _parse_image(data["left_image"])

        inputs = {
            "state": raw_state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_ if self.model_type != _model.ModelType.PI0_FAST else np.True_,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class PikaRM65Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :7]}
