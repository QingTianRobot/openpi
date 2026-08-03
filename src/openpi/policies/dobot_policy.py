import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_dobot_example() -> dict:
    """Create an inference observation in the Dobot runtime schema."""
    return {
        "state": np.zeros(13, dtype=np.float32),
        "global_image": np.zeros((480, 640, 3), dtype=np.uint8),
        "wrist_image": np.zeros((530, 848, 3), dtype=np.uint8),
        "prompt": "pick up the tape roll",
    }


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DobotInputs(transforms.DataTransformFn):
    """Map Dobot Nova2 observations to the pi0/pi0.5 model schema."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        raw_state = np.asarray(data["state"])
        if raw_state.shape[-1] != 13:
            raise ValueError(f"Expected 13-dimensional Dobot state, got {raw_state.shape}")

        # Six arm joint positions plus the gripper opening. TCP pose is omitted
        # because the recorded action space is six Cartesian jog commands plus gripper.
        state = np.concatenate([raw_state[..., :6], raw_state[..., -1:]], axis=-1)
        base_image = _parse_image(data["global_image"])
        wrist_image = _parse_image(data["wrist_image"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class DobotOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :7]}
