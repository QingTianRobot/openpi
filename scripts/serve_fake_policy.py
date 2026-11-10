"""Pipeline smoke-test policy server for the RealMan arm. It never moves the robot.

It speaks the normal OpenPI websocket protocol (`WebsocketPolicyServer`) but needs no model, no jax and no GPU, so it
can run on the robot host. Every request is checked against the observation schema policy_bridge sends, then answered
with an action chunk of zero Cartesian velocity. The gripper dimension holds the gripper's current opening instead of
0, because 0 means "close" for policy_bridge.

    PYTHONPATH=src:packages/openpi-client/src python scripts/serve_fake_policy.py --port 18000 \\
        --prompt "pick up the power adapter and put it in the blue bin"
"""

import argparse
import logging
import time

import numpy as np
from openpi_client import base_policy as _base_policy

from openpi.serving import websocket_policy_server

STATE_DIM = 7
ACTION_DIM = 7


class ObservationError(ValueError):
    """The observation does not match what policy_bridge is expected to send."""


class FakeRealmanPolicy(_base_policy.BasePolicy):
    def __init__(self, *, action_horizon: int = 16, expected_prompt: str | None = None, log_every_s: float = 5.0):
        self._horizon = action_horizon
        self._expected_prompt = expected_prompt
        self._log_every_s = log_every_s
        self._count = 0
        self._window_start = time.monotonic()
        self._window_count = 0

    def validate(self, obs: dict) -> None:
        for key in ("state", "global_image", "wrist_image", "prompt"):
            if key not in obs:
                raise ObservationError(f"missing key {key!r}; got {sorted(obs)}")
        state = np.asarray(obs["state"])
        if state.shape != (STATE_DIM,) or not np.issubdtype(state.dtype, np.floating):
            raise ObservationError(f"state must be a float ({STATE_DIM},) vector, got {state.dtype} {state.shape}")
        if not np.isfinite(state).all():
            raise ObservationError("state contains NaN or inf")
        if not 0.0 <= float(state[6]) <= 1.0:
            raise ObservationError(f"state[6] (gripper opening) must be in [0, 1], got {float(state[6])}")
        for key in ("global_image", "wrist_image"):
            image = np.asarray(obs[key])
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ObservationError(f"{key} must be a uint8 (H, W, 3) image, got {image.dtype} {image.shape}")
            if image.shape[0] < 32 or image.shape[1] < 32:
                raise ObservationError(f"{key} is implausibly small: {image.shape}")
        if not isinstance(obs["prompt"], str) or not obs["prompt"]:
            raise ObservationError("prompt must be a non-empty string")
        if self._expected_prompt is not None and obs["prompt"] != self._expected_prompt:
            raise ObservationError(f"prompt {obs['prompt']!r} differs from the expected {self._expected_prompt!r}")

    def infer(self, obs: dict) -> dict:
        self.validate(obs)
        actions = np.zeros((self._horizon, ACTION_DIM), dtype=np.float32)
        actions[:, 6] = float(np.asarray(obs["state"])[6])  # hold the gripper where it is
        self._log(obs)
        return {"actions": actions}

    def _log(self, obs: dict) -> None:
        self._count += 1
        self._window_count += 1
        if self._count == 1:
            logging.info(
                "first observation OK: state=%s global=%s wrist=%s prompt=%r",
                np.round(np.asarray(obs["state"]), 3).tolist(),
                np.asarray(obs["global_image"]).shape,
                np.asarray(obs["wrist_image"]).shape,
                obs["prompt"],
            )
        elapsed = time.monotonic() - self._window_start
        if elapsed >= self._log_every_s:
            logging.info(
                "%d requests total, %.1f/s over the last %.0fs", self._count, self._window_count / elapsed, elapsed
            )
            self._window_start, self._window_count = time.monotonic(), 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Loopback by default: this server answers anyone who connects.
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument(
        "--prompt",
        default=None,
        help="Reject requests whose prompt differs, which catches a mismatch with the training data.",
    )
    args = parser.parse_args(argv)
    policy = FakeRealmanPolicy(action_horizon=args.action_horizon, expected_prompt=args.prompt)
    metadata = {"policy": "fake-realman-zero-velocity", "action_horizon": args.action_horizon, "action_dim": ACTION_DIM}
    logging.info("fake policy listening on %s:%d (zero velocity, gripper held)", args.host, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port, metadata=metadata
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    main()
