import socket
import threading
import time

import numpy as np
from openpi_client import websocket_client_policy
import pytest

from openpi.serving import websocket_policy_server
from scripts import serve_fake_policy as fake

PROMPT = "pick up the power adapter and put it in the blue bin"


def _obs(**overrides):
    obs = {
        "state": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75], dtype=np.float32),
        "global_image": np.zeros((480, 640, 3), dtype=np.uint8),
        "wrist_image": np.zeros((480, 640, 3), dtype=np.uint8),
        "prompt": PROMPT,
    }
    obs.update(overrides)
    return obs


def test_zero_velocity_and_gripper_hold():
    actions = fake.FakeRealmanPolicy().infer(_obs())["actions"]

    assert actions.shape == (16, 7)
    assert actions.dtype == np.float32
    assert not actions[:, :6].any()
    np.testing.assert_allclose(actions[:, 6], 0.75)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"state": np.zeros(6, dtype=np.float32)}, "state must be"),
        ({"state": np.zeros(7, dtype=np.int64)}, "state must be"),
        ({"state": np.array([0, 0, 0, 0, 0, 0, np.nan], dtype=np.float32)}, "NaN"),
        ({"state": np.array([0, 0, 0, 0, 0, 0, 1.5], dtype=np.float32)}, "gripper opening"),
        ({"global_image": np.zeros((480, 640), dtype=np.uint8)}, "global_image"),
        ({"wrist_image": np.zeros((480, 640, 3), dtype=np.float32)}, "wrist_image"),
        ({"prompt": ""}, "prompt"),
    ],
)
def test_bad_observations_are_rejected(overrides, message):
    with pytest.raises(fake.ObservationError, match=message):
        fake.FakeRealmanPolicy().infer(_obs(**overrides))


def test_missing_key_is_rejected():
    obs = _obs()
    del obs["wrist_image"]
    with pytest.raises(fake.ObservationError, match="missing key"):
        fake.FakeRealmanPolicy().infer(obs)


def test_expected_prompt_mismatch_is_rejected():
    policy = fake.FakeRealmanPolicy(expected_prompt=PROMPT)
    policy.infer(_obs())
    with pytest.raises(fake.ObservationError, match="differs"):
        policy.infer(_obs(prompt="pick the red block"))


def test_websocket_round_trip():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = websocket_policy_server.WebsocketPolicyServer(
        fake.FakeRealmanPolicy(expected_prompt=PROMPT), host="127.0.0.1", port=port, metadata={"policy": "fake"}
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    client = None
    for _ in range(50):
        try:
            client = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=port)
            break
        except Exception:
            time.sleep(0.1)
    assert client is not None

    result = client.infer(_obs())

    assert result["actions"].shape == (16, 7)
    assert not result["actions"][:, :6].any()
    assert client.get_server_metadata()["policy"] == "fake"
