import numpy as np

from openpi.models import model as _model
from openpi.policies import pika_rm65_policy


def test_pika_rm65_inputs_accept_inference_observation_without_actions():
    transformed = pika_rm65_policy.PikaRM65Inputs(_model.ModelType.PI05)(pika_rm65_policy.make_pika_rm65_example())

    assert transformed["state"].shape == (7,)
    assert transformed["image"]["base_0_rgb"].shape == (360, 640, 3)
    assert transformed["image"]["left_wrist_0_rgb"].shape == (360, 640, 3)
    assert transformed["prompt"] == "pick and place"
    assert "actions" not in transformed


def test_pika_rm65_inputs_preserve_training_actions():
    example = pika_rm65_policy.make_pika_rm65_example()
    example["actions"] = np.zeros((50, 7), dtype=np.float32)

    transformed = pika_rm65_policy.PikaRM65Inputs(_model.ModelType.PI05)(example)

    assert transformed["actions"].shape == (50, 7)


def test_pika_rm65_outputs_slice_padded_actions():
    transformed = pika_rm65_policy.PikaRM65Outputs()({"actions": np.zeros((50, 32), dtype=np.float32)})

    assert transformed["actions"].shape == (50, 7)
