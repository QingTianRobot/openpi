import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import realman_policy


def test_inputs_accept_inference_observation_without_actions():
    transformed = realman_policy.RealmanInputs(_model.ModelType.PI05)(realman_policy.make_realman_example())

    assert transformed["state"].shape == (7,)
    assert transformed["image"]["base_0_rgb"].shape == (480, 640, 3)
    assert transformed["image"]["left_wrist_0_rgb"].shape == (480, 640, 3)
    assert transformed["image"]["right_wrist_0_rgb"].shape == (480, 640, 3)
    assert transformed["prompt"] == "pick_and_place"
    assert "actions" not in transformed


def test_inputs_preserve_training_actions():
    example = realman_policy.make_realman_example()
    example["actions"] = np.ones((16, 7), dtype=np.float32)

    transformed = realman_policy.RealmanInputs(_model.ModelType.PI05)(example)

    assert transformed["actions"].shape == (16, 7)


def test_missing_wrist_camera_is_zero_padded_and_masked():
    example = realman_policy.make_realman_example()
    example["global_image"] = np.full((480, 640, 3), 7, dtype=np.uint8)

    transformed = realman_policy.RealmanInputs(_model.ModelType.PI05)(example)

    assert not transformed["image"]["right_wrist_0_rgb"].any()
    assert transformed["image_mask"]["base_0_rgb"]
    assert transformed["image_mask"]["left_wrist_0_rgb"]
    assert not transformed["image_mask"]["right_wrist_0_rgb"]
    # pi0-FAST does not mask padding images.
    fast = realman_policy.RealmanInputs(_model.ModelType.PI0_FAST)(example)
    assert fast["image_mask"]["right_wrist_0_rgb"]


def test_chw_uint8_and_float_images_become_hwc_uint8():
    example = realman_policy.make_realman_example()
    example["global_image"] = np.zeros((3, 480, 640), dtype=np.uint8)  # what the dataset loader returns
    example["wrist_image"] = np.full((480, 640, 3), 0.5, dtype=np.float32)

    transformed = realman_policy.RealmanInputs(_model.ModelType.PI05)(example)

    assert transformed["image"]["base_0_rgb"].shape == (480, 640, 3)
    assert transformed["image"]["left_wrist_0_rgb"].dtype == np.uint8
    assert transformed["image"]["left_wrist_0_rgb"][0, 0, 0] == 127


def test_wrong_dimensions_raise():
    example = realman_policy.make_realman_example()
    example["state"] = np.zeros(13, dtype=np.float32)
    with pytest.raises(ValueError, match="7-dimensional"):
        realman_policy.RealmanInputs(_model.ModelType.PI05)(example)

    example = realman_policy.make_realman_example()
    example["actions"] = np.zeros((16, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="7-dimensional"):
        realman_policy.RealmanInputs(_model.ModelType.PI05)(example)


def test_outputs_drop_padding_dimensions():
    chunk = np.arange(16 * 32, dtype=np.float32).reshape(16, 32)

    out = realman_policy.RealmanOutputs()({"actions": chunk})

    assert out["actions"].shape == (16, 7)
    np.testing.assert_array_equal(out["actions"], chunk[:, :7])
