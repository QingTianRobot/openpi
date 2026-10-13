"""Build the training view of a (cleaned) RealMan LeRobot v3 recording for ONE arm.

The recording stores three arms with canonical, model-independent features. The pi0.5 pipeline needs two plain
columns instead, so this script writes a new dataset directory next to the recording with

  observation.state  (7)  right arm joints (rad) + gripper opening 0..1 (0 = closed, 1 = open)
  action             (7)  6 Cartesian velocities divided by the scale policy_bridge multiplies back
                          (linear / 0.25 m/s, angular / 0.6 rad/s, clipped to [-1, 1]) + gripper target (0 or 1)

The recording has no gripper command. The target is derived from the *future* gripper readings: the gripper
closes slowly (about 1.5 s) and opens quickly (about 0.4 s), so a fixed look-ahead would release far too early.
Instead each closing / opening event is located and the target flips where the gripper starts to move, shifted
earlier by `command_latency_frames` to account for the delay between command and motion.

Videos are hard-linked (not symlinked: a symlink to a host path dangles inside a container), so the view needs no
extra disk space and must live on the same filesystem as the recording. The recording is never modified.

Example:
    uv run scripts/make_realman_view.py --src ~/data/realman/realman__pi05-three-arm \\
        --dst ~/data/realman/realman_right_arm_view
"""

import dataclasses
import datetime
import json
import logging
import os
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tyro

ARM_INDEX = {"l": 0, "m": 1, "r": 2}
QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}
DEFAULT_PROMPT = "pick up the power adapter and put it in the blue bin"


@dataclasses.dataclass(frozen=True)
class ViewParams:
    arm: str = "r"
    # Must equal policy_bridge's `downlink.velocity.twist_scale`: it multiplies the policy output back to SI units.
    linear_scale: float = 0.25  # m/s
    angular_scale: float = 0.6  # rad/s
    # Gripper reading at fully open / fully closed. Defaults: right gripper, config/ros/gripper.yaml.
    gripper_open_raw: float = 50.0
    gripper_close_raw: float = 8500.0
    # Readings above this count as closed. None means halfway between open and closed. A gripper that closes on an
    # object stalls well below `gripper_close_raw` (about 5700 here), so do not set this close to it.
    closed_threshold_raw: float | None = None
    # How many frames before the gripper starts to move the target flips. Not measured: 5 frames is 0.17 s.
    command_latency_frames: int = 5
    # Median filter window that removes single-frame reading glitches before events are located.
    smoothing_window: int = 5
    prompt: str = DEFAULT_PROMPT
    cameras: tuple[str, ...] = ("d435", "orbbec-right")


def gripper_open_fraction(raw: np.ndarray, open_raw: float, close_raw: float) -> np.ndarray:
    """Raw gripper reading -> 0 (closed) .. 1 (open), the convention of policy_bridge and gripper_manager."""
    fraction = (close_raw - np.asarray(raw, dtype=np.float64)) / (close_raw - open_raw)
    return np.clip(fraction, 0.0, 1.0).astype(np.float32)


def _median_filter(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    padded = np.pad(values, window // 2, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, window), axis=1)


def derive_gripper_target(
    raw: np.ndarray, params: ViewParams, *, lookback: int = 90, margin: float = 100.0
) -> np.ndarray:
    """Per-frame gripper target (0 = close, 1 = open) from the readings of the whole episode."""
    smooth = _median_filter(np.asarray(raw, dtype=np.float32), params.smoothing_window)
    threshold = (
        (params.gripper_open_raw + params.gripper_close_raw) / 2
        if params.closed_threshold_raw is None
        else params.closed_threshold_raw
    )
    closed = smooth > threshold
    commanded_closed = np.full(len(smooth), closed[0])
    previous_crossing = 0
    for crossing in np.flatnonzero(closed[1:] != closed[:-1]) + 1:
        to_closed = bool(closed[crossing])
        before = smooth[max(previous_crossing, crossing - lookback) : crossing]
        rest = before.min() if to_closed else before.max()  # where the gripper sat before this move
        start = crossing
        while start > previous_crossing and (
            smooth[start - 1] > rest + margin if to_closed else smooth[start - 1] < rest - margin
        ):
            start -= 1
        start = max(previous_crossing, start - params.command_latency_frames)
        commanded_closed[start:] = to_closed
        previous_crossing = crossing
    return np.where(commanded_closed, 0.0, 1.0).astype(np.float32)


def make_state_action(
    joint_position: np.ndarray, gripper_position: np.ndarray, command_action: np.ndarray, params: ViewParams
) -> tuple[np.ndarray, np.ndarray]:
    """One episode: canonical (T, 18) / (T, 3) / (T, 18) arrays -> state (T, 7) and action (T, 7)."""
    arm = ARM_INDEX[params.arm]
    joints = np.asarray(joint_position, dtype=np.float32)[:, arm * 6 : arm * 6 + 6]
    raw = np.asarray(gripper_position, dtype=np.float32)[:, arm]
    state = np.concatenate(
        [joints, gripper_open_fraction(raw, params.gripper_open_raw, params.gripper_close_raw)[:, None]], axis=1
    )
    velocity = np.asarray(command_action, dtype=np.float32)[:, arm * 6 : arm * 6 + 6].copy()
    velocity[:, :3] /= params.linear_scale
    velocity[:, 3:] /= params.angular_scale
    action = np.concatenate([np.clip(velocity, -1.0, 1.0), derive_gripper_target(raw, params)[:, None]], axis=1)
    return state.astype(np.float32), action.astype(np.float32)


def _feature_stats(values: np.ndarray) -> dict[str, list[float]]:
    stats = {
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "count": np.array([len(values)]),
    }
    stats.update({name: np.quantile(values, q, axis=0) for name, q in QUANTILES.items()})
    return {k: np.asarray(v, dtype=np.float64).tolist() for k, v in stats.items()}


def _list_column(values: np.ndarray) -> pa.Array:
    return pa.array(values.tolist(), type=pa.list_(pa.float32()))


def build_view(src: pathlib.Path, dst: pathlib.Path, params: ViewParams) -> dict:
    if params.arm not in ARM_INDEX:
        raise ValueError(f"arm must be one of {sorted(ARM_INDEX)}")
    if dst.exists():
        raise FileExistsError(f"{dst} already exists; refusing to overwrite")
    info = json.loads((src / "meta" / "info.json").read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected a LeRobot v3.0 recording, got {info.get('codebase_version')!r}")
    fps = info["fps"]
    for camera in params.cameras:
        if f"observation.images.{camera}" not in info["features"]:
            raise ValueError(f"camera {camera!r} is not in the recording: {sorted(info['features'])}")

    episode_rows = []
    for path in sorted((src / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(path)
        if table.num_rows != 1:
            raise ValueError(f"{path} must hold exactly one episode")
        episode_rows.append(table.to_pylist()[0])
    episode_rows.sort(key=lambda r: r["episode_index"])
    if [r["episode_index"] for r in episode_rows] != list(range(info["total_episodes"])):
        raise ValueError("episode indices must be 0..total_episodes-1 (clean the recording first)")

    for sub in ("meta/episodes/chunk-000", "data/chunk-000"):
        (dst / sub).mkdir(parents=True)
    all_states, all_actions, new_rows = [], [], []
    for row in episode_rows:
        episode, length = row["episode_index"], row["length"]
        source_data = (
            src / "data" / f"chunk-{row['data/chunk_index']:03d}" / f"file-{row['data/file_index']:03d}.parquet"
        )
        table = pq.read_table(
            source_data,
            columns=["observation.joint_position", "observation.gripper_position", "action.command_action"],
        )
        if table.num_rows != length:
            raise ValueError(f"{source_data}: {table.num_rows} rows, metadata says {length}")
        state, action = make_state_action(
            np.array(table["observation.joint_position"].to_pylist(), dtype=np.float32),
            np.array(table["observation.gripper_position"].to_pylist(), dtype=np.float32),
            np.array(table["action.command_action"].to_pylist(), dtype=np.float32),
            params,
        )
        index = np.arange(row["dataset_from_index"], row["dataset_to_index"], dtype=np.int64)
        pq.write_table(
            pa.table(
                {
                    "observation.state": _list_column(state),
                    "action": _list_column(action),
                    "task_index": pa.array(np.zeros(length, dtype=np.int64)),
                    "episode_index": pa.array(np.full(length, episode, dtype=np.int64)),
                    "frame_index": pa.array(np.arange(length, dtype=np.int64)),
                    "index": pa.array(index),
                    "timestamp": pa.array((np.arange(length) / fps).astype(np.float32)),
                }
            ),
            dst / "data" / "chunk-000" / f"file-{episode:03d}.parquet",
            compression="snappy",
        )
        all_states.append(state)
        all_actions.append(action)

        new = {
            k: row[k]
            for k in (
                "episode_index",
                "length",
                "data/chunk_index",
                "data/file_index",
                "dataset_from_index",
                "dataset_to_index",
                "meta/episodes/chunk_index",
                "meta/episodes/file_index",
            )
        }
        new["tasks"] = [params.prompt]
        for camera in params.cameras:
            base = f"videos/observation.images.{camera}/"
            for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
                new[base + field] = row[base + field]
            source_video = src / "videos" / f"observation.images.{camera}" / f"chunk-{row[base + 'chunk_index']:03d}"
            source_video = source_video / f"file-{row[base + 'file_index']:03d}.mp4"
            target = dst / "videos" / f"observation.images.{camera}" / f"chunk-{row[base + 'chunk_index']:03d}"
            target.mkdir(parents=True, exist_ok=True)
            os.link(source_video, target / source_video.name)
        new_rows.append(new)
    for new in new_rows:
        pq.write_table(
            pa.Table.from_pylist([new]),
            dst / "meta" / "episodes" / "chunk-000" / f"file-{new['episode_index']:03d}.parquet",
            compression="snappy",
        )

    names_joint = [f"{params.arm}.joint_{i}" for i in range(1, 7)]
    features = {
        "observation.state": {"dtype": "float32", "shape": [7], "names": [*names_joint, "gripper_open_fraction"]},
        "action": {
            "dtype": "float32",
            "shape": [7],
            "names": ["vx", "vy", "vz", "wx", "wy", "wz", "gripper_target"],
        },
        **{f"observation.images.{c}": info["features"][f"observation.images.{c}"] for c in params.cameras},
        **{k: info["features"][k] for k in ("timestamp", "frame_index", "episode_index", "index", "task_index")},
    }
    (dst / "meta" / "info.json").write_text(json.dumps({**info, "features": features}, indent=4))
    pq.write_table(pa.table({"task_index": [0], "task": [params.prompt]}), dst / "meta" / "tasks.parquet")
    states, actions = np.concatenate(all_states), np.concatenate(all_actions)
    (dst / "meta" / "stats.json").write_text(
        json.dumps({"observation.state": _feature_stats(states), "action": _feature_stats(actions)}, indent=4)
    )
    contract = {
        "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "source": str(src),
        "script": "scripts/make_realman_view.py",
        "arm": params.arm,
        "episodes": info["total_episodes"],
        "frames": info["total_frames"],
        "fps": fps,
        "prompt": params.prompt,
        "state": "6 joint positions (rad) + gripper open fraction (0 = closed, 1 = open)",
        "action": "6 Cartesian velocities / scale, clipped to [-1, 1] + gripper target (0 = close, 1 = open)",
        "params": dataclasses.asdict(params),
        "gripper_target": "derived from the future readings of the whole episode, see derive_gripper_target",
        "normalizer_asset_version": "norm_stats computed by compute_norm_stats for repo_id local/realman_right_arm",
    }
    (dst / "contract.json").write_text(json.dumps(contract, indent=2))
    logging.info("view written to %s: %d episodes, %d frames", dst, len(new_rows), len(states))
    return contract


@dataclasses.dataclass
class Args:
    src: pathlib.Path
    dst: pathlib.Path
    arm: str = "r"
    prompt: str = DEFAULT_PROMPT
    linear_scale: float = 0.25
    angular_scale: float = 0.6
    command_latency_frames: int = 5


def main(args: Args) -> None:
    params = ViewParams(
        arm=args.arm,
        prompt=args.prompt,
        linear_scale=args.linear_scale,
        angular_scale=args.angular_scale,
        command_latency_frames=args.command_latency_frames,
    )
    build_view(args.src.expanduser(), args.dst.expanduser(), params)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
