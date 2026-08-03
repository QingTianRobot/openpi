import numpy as np

from openpi.models import model as _model
from openpi.policies import dobot_policy


def test_dobot_inputs_accept_inference_observation_without_actions():
    transformed = dobot_policy.DobotInputs(_model.ModelType.PI05)(dobot_policy.make_dobot_example())

    assert transformed["state"].shape == (7,)
    assert transformed["image"]["base_0_rgb"].shape == (480, 640, 3)
    assert transformed["image"]["left_wrist_0_rgb"].shape == (530, 848, 3)
    assert transformed["prompt"] == "pick up the tape roll"
    assert "actions" not in transformed


def test_dobot_inputs_preserve_training_actions():
    example = dobot_policy.make_dobot_example()
    example["actions"] = np.zeros((16, 7), dtype=np.float32)

    transformed = dobot_policy.DobotInputs(_model.ModelType.PI05)(example)

    assert transformed["actions"].shape == (16, 7)
