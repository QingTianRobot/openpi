import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.training import lerobot_v3_dataset as v3
from openpi.training import lerobot_v3_fixture as fx
from scripts import make_realman_view as mv

pytest.importorskip("av")

NO_LATENCY = mv.ViewParams(command_latency_frames=0)


def _close_then_open_curve() -> np.ndarray:
    """Open until frame 20, slow close (frames 20-59), hold, fast open (frames 90-99), open until the end."""
    raw = np.full(120, 50.0)
    raw[20:60] = np.linspace(50, 5800, 40)
    raw[60:90] = 5800
    raw[90:100] = np.linspace(5800, 50, 10)
    return raw


def test_gripper_open_fraction():
    fraction = mv.gripper_open_fraction(np.array([50.0, 8500.0, 4275.0, -5.0, 9000.0]), 50.0, 8500.0)
    np.testing.assert_allclose(fraction, [1.0, 0.0, 0.5, 1.0, 0.0], atol=1e-6)


def test_target_flips_where_the_gripper_starts_to_move():
    target = mv.derive_gripper_target(_close_then_open_curve(), NO_LATENCY)

    # Closing: the reading first leaves the open level at frame 21. Opening: it first drops below the plateau at 91.
    assert (target[:21] == 1).all()
    assert (target[21:91] == 0).all()
    assert (target[91:] == 1).all()


def test_command_latency_moves_the_flips_earlier():
    target = mv.derive_gripper_target(_close_then_open_curve(), mv.ViewParams(command_latency_frames=5))

    assert (target[:16] == 1).all()
    assert (target[16:86] == 0).all()
    assert (target[86:] == 1).all()


def test_closing_is_not_shifted_by_a_fixed_lookahead():
    # The slow close takes ~30 frames to cross the threshold and the open ~1: the open flip must stay at frame 91,
    # not 30 frames earlier as a constant look-ahead tuned on the close would put it.
    target = mv.derive_gripper_target(_close_then_open_curve(), NO_LATENCY)
    assert target[90] == 0


def test_single_frame_glitch_is_ignored():
    raw = np.full(60, 50.0)
    raw[30] = 8000.0

    assert (mv.derive_gripper_target(raw, NO_LATENCY) == 1).all()


def test_episode_that_starts_closed_and_one_that_never_closes():
    starts_closed = np.full(60, 50.0)
    starts_closed[:15] = 5800
    starts_closed[15:25] = np.linspace(5800, 50, 10)
    target = mv.derive_gripper_target(starts_closed, NO_LATENCY)
    assert (target[:16] == 0).all()
    assert (target[16:] == 1).all()

    assert (mv.derive_gripper_target(np.full(40, 50.0), NO_LATENCY) == 1).all()


def test_episode_that_ends_closed_stays_closed():
    raw = _close_then_open_curve()[:80]

    target = mv.derive_gripper_target(raw, NO_LATENCY)

    assert (target[:21] == 1).all()
    assert (target[21:] == 0).all()


def test_flip_is_not_moved_before_the_previous_event():
    raw = np.full(60, 50.0)
    raw[3:20] = np.linspace(50, 5800, 17)  # starts moving almost immediately

    target = mv.derive_gripper_target(raw, mv.ViewParams(command_latency_frames=10))

    assert target[0] == 0  # clamped to the start of the episode instead of wrapping around
    assert target.shape == (60,)


def test_state_and_action_for_the_right_arm():
    length = 4
    joints = np.arange(length * 18, dtype=np.float32).reshape(length, 18)
    grip = np.zeros((length, 3), dtype=np.float32)
    grip[:, 2] = [50.0, 50.0, 8500.0, 8500.0]
    command = np.zeros((length, 18), dtype=np.float32)
    command[:, 12:18] = [0.125, 0.5, -0.5, 0.3, 1.2, 0.0]  # right arm; the other arms stay idle

    state, action = mv.make_state_action(joints, grip, command, mv.ViewParams())

    assert state.shape == (length, 7)
    assert action.shape == (length, 7)
    np.testing.assert_array_equal(state[:, :6], joints[:, 12:18])
    np.testing.assert_allclose(state[:, 6], [1.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(action[0, :6], [0.5, 1.0, -1.0, 0.5, 1.0, 0.0], atol=1e-6)  # /0.25, /0.6, clipped
    assert set(np.unique(action[:, 6])) <= {0.0, 1.0}


def _make_recording(root, lengths=(100, 90, 80), tasks=("old task name",) * 3):
    cameras = ("d435", "orbbec-right", "orbbec-left")
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    video_feature = {"dtype": "video", "shape": [fx.HEIGHT, fx.WIDTH, 3], "names": ["height", "width", "channels"]}
    scalar = {"dtype": "int64", "shape": [1], "names": None}
    info = {
        "codebase_version": "v3.0",
        "fps": fx.FPS,
        "total_episodes": len(lengths),
        "total_frames": sum(lengths),
        "splits": {"train": f"0:{len(lengths)}"},
        "features": {
            **{f"observation.images.{c}": video_feature for c in cameras},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            **dict.fromkeys(("frame_index", "episode_index", "index", "task_index"), scalar),
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    pq.write_table(pa.table({"task_index": [0], "task": ["old task name"]}), root / "meta" / "tasks.parquet")
    curves = [
        _close_then_open_curve()[:100],
        np.concatenate([np.full(15, 5800.0), np.linspace(5800, 50, 10), np.full(65, 50.0)]),
        np.full(80, 50.0),
    ]
    start = 0
    for episode, length in enumerate(lengths):
        frames = np.arange(length)
        joints = (100 * episode + np.arange(18))[None, :] + 0.01 * frames[:, None]
        grip = np.zeros((length, 3))
        grip[:, 2] = curves[episode]
        command = np.zeros((length, 18))
        command[:, 12:18] = [0.125, 0.5, -0.5, 0.3, 1.2, 0.0]
        data_dir = root / "data" / "chunk-000"
        data_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "observation.joint_position": pa.array(joints.astype(np.float32).tolist(), pa.list_(pa.float32())),
                    "observation.gripper_position": pa.array(grip.astype(np.float32).tolist(), pa.list_(pa.float32())),
                    "action.command_action": pa.array(command.astype(np.float32).tolist(), pa.list_(pa.float32())),
                    "episode_index": pa.array(np.full(length, episode, dtype=np.int64)),
                }
            ),
            data_dir / f"file-{episode:03d}.parquet",
        )
        row = {
            "episode_index": episode,
            "length": length,
            "data/chunk_index": 0,
            "data/file_index": episode,
            "dataset_from_index": start,
            "dataset_to_index": start + length,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": episode,
            "tasks": [tasks[episode]],
        }
        for camera in cameras:
            base = f"videos/observation.images.{camera}/"
            row |= {base + "chunk_index": 0, base + "file_index": episode, base + "from_timestamp": 0.0}
            row[base + "to_timestamp"] = length / fx.FPS
            fx.write_video(
                root / "videos" / f"observation.images.{camera}" / "chunk-000" / f"file-{episode:03d}.mp4",
                [fx.gray_level_d435(t % 20) for t in range(length)],
            )
        pq.write_table(
            pa.Table.from_pylist([row]), root / "meta" / "episodes" / "chunk-000" / f"file-{episode:03d}.parquet"
        )
        start += length
    return root


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    base = tmp_path_factory.mktemp("view")
    src = _make_recording(base / "recording")
    dst = base / "view"
    contract = mv.build_view(src, dst, mv.ViewParams())
    return src, dst, contract


def test_view_files_and_metadata(built):
    src, dst, contract = built

    info = json.loads((dst / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 3
    assert info["total_frames"] == 270
    assert {"observation.state", "action"} <= set(info["features"])
    assert "observation.images.orbbec-left" not in info["features"]  # not used by this arm
    assert pq.read_table(dst / "meta" / "tasks.parquet").to_pylist()[0]["task"] == mv.DEFAULT_PROMPT
    stats = json.loads((dst / "meta" / "stats.json").read_text())
    assert stats["observation.state"]["count"] == [270]
    assert len(stats["action"]["q99"]) == 7
    assert contract["arm"] == "r"
    assert (dst / "contract.json").exists()
    assert not (dst / "videos" / "observation.images.orbbec-left").exists()


def test_videos_are_hard_links_not_symlinks(built):
    src, dst, _ = built
    for camera in ("d435", "orbbec-right"):
        source = src / "videos" / f"observation.images.{camera}" / "chunk-000" / "file-001.mp4"
        linked = dst / "videos" / f"observation.images.{camera}" / "chunk-000" / "file-001.mp4"
        assert not linked.is_symlink()
        assert os.stat(source).st_ino == os.stat(linked).st_ino


def test_view_is_readable_by_the_loader_and_has_the_derived_values(built):
    _, dst, _ = built
    dataset = v3.LocalLeRobotV3Dataset(
        dst,
        action_horizon=4,
        video_keys={
            "observation.images.global": "observation.images.d435",
            "observation.images.wrist": "observation.images.orbbec-right",
        },
    )

    assert len(dataset) == 270
    sample = dataset[dataset.indices_for(0)[40]]
    np.testing.assert_allclose(sample["observation.state"][:6], 12 + np.arange(6) + 0.4, atol=1e-4)
    assert sample["prompt"] == mv.DEFAULT_PROMPT
    assert sample["observation.images.global"].shape == (3, fx.HEIGHT, fx.WIDTH)
    assert sample["observation.images.wrist"].shape == (3, fx.HEIGHT, fx.WIDTH)
    np.testing.assert_allclose(sample["action"][0, :6], [0.5, 1.0, -1.0, 0.5, 1.0, 0.0], atol=1e-6)

    # Episode 0 closes at frame 21 and opens at 91, both shifted 5 frames earlier by the default latency.
    target = np.array([dataset[i]["action"][0, 6] for i in dataset.indices_for(0)])
    assert (target[:16] == 1).all()
    assert (target[16:86] == 0).all()
    assert (target[86:] == 1).all()
    # Episode 1 starts closed, opens at frame 16; episode 2 never closes.
    target1 = np.array([dataset[i]["action"][0, 6] for i in dataset.indices_for(1)])
    assert (target1[:11] == 0).all()
    assert (target1[11:] == 1).all()
    assert all(dataset[i]["action"][0, 6] == 1 for i in dataset.indices_for(2))


def test_existing_destination_and_unknown_camera_are_rejected(built, tmp_path):
    src, dst, _ = built
    with pytest.raises(FileExistsError):
        mv.build_view(src, dst, mv.ViewParams())
    with pytest.raises(ValueError, match="camera"):
        mv.build_view(src, tmp_path / "other", mv.ViewParams(cameras=("d435", "nope")))


def test_multi_task_view_maps_each_recording_task_to_its_prompt(tmp_path):
    src = _make_recording(tmp_path / "recording", tasks=("pick_and_place", "pick_only", "pick_and_place"))
    dst = tmp_path / "view"
    pairs = (("pick_and_place", "place it"), ("pick_only", "pick it"))
    contract = mv.build_view(src, dst, mv.ViewParams(task_prompts=pairs))

    assert pq.read_table(dst / "meta" / "tasks.parquet").to_pylist() == [
        {"task_index": 0, "task": "place it"},
        {"task_index": 1, "task": "pick it"},
    ]
    assert json.loads((dst / "meta" / "info.json").read_text())["total_tasks"] == 2
    for episode, (task_index, prompt) in enumerate([(0, "place it"), (1, "pick it"), (0, "place it")]):
        data = pq.read_table(dst / "data" / "chunk-000" / f"file-{episode:03d}.parquet")
        assert set(data["task_index"].to_pylist()) == {task_index}
        row = pq.read_table(dst / "meta" / "episodes" / "chunk-000" / f"file-{episode:03d}.parquet").to_pylist()[0]
        assert row["tasks"] == [prompt]
    assert contract["prompts"] == {"pick_and_place": "place it", "pick_only": "pick it"}


def test_multi_task_view_rejects_an_unmapped_task(tmp_path):
    src = _make_recording(tmp_path / "recording", tasks=("pick_and_place", "pick_only", "pick_and_place"))
    with pytest.raises(ValueError, match="no entry in task_prompts"):
        mv.build_view(src, tmp_path / "view", mv.ViewParams(task_prompts=(("pick_and_place", "place it"),)))


def test_task_prompt_without_equals_raises_value_error(tmp_path):
    src = _make_recording(tmp_path / "recording")
    with pytest.raises(ValueError, match="--task-prompt value must be"):
        mv.main(mv.Args(src=src, dst=tmp_path / "view", task_prompt=("pick_only",)))
