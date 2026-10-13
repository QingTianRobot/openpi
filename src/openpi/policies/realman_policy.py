import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# One RealMan RM65 arm: 6 joint positions + gripper opening (0 = closed, 1 = open) in, and
# 6 normalized Cartesian velocity commands + gripper target out. The model works in 32 dims internally.
REALMAN_STATE_DIM = 7
REALMAN_ACTION_DIM = 7


def make_realman_example() -> dict:
    """Create an inference observation in the RealMan runtime schema (the keys policy_bridge sends)."""
    return {
        "state": np.zeros(REALMAN_STATE_DIM, dtype=np.float32),
        "global_image": np.zeros((480, 640, 3), dtype=np.uint8),
        "wrist_image": np.zeros((480, 640, 3), dtype=np.uint8),
        "prompt": "pick_and_place",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RealmanInputs(transforms.DataTransformFn):
    """Map RealMan observations to the pi0/pi0.5 model schema."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape[-1] != REALMAN_STATE_DIM:
            raise ValueError(f"Expected {REALMAN_STATE_DIM}-dimensional RealMan state, got {state.shape}")
        global_image = _parse_image(data["global_image"])
        wrist_image = _parse_image(data["wrist_image"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": global_image,
                "left_wrist_0_rgb": wrist_image,
                # There is no second wrist camera: pad with zeros and, for pi0 / pi0.5, mask it out.
                "right_wrist_0_rgb": np.zeros_like(global_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Padding images are only masked for pi0 / pi0.5, not pi0-FAST. Do not change this.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != REALMAN_ACTION_DIM:
                raise ValueError(f"Expected {REALMAN_ACTION_DIM}-dimensional RealMan actions, got {actions.shape}")
            inputs["actions"] = actions
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class RealmanOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # The model predicts 32 dims; only the first 7 are meaningful for one RealMan arm.
        return {"actions": np.asarray(data["actions"])[..., :REALMAN_ACTION_DIM]}
