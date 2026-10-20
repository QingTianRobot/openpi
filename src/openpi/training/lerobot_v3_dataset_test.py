import dataclasses

import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import lerobot_v3_dataset as v3
from openpi.training import lerobot_v3_fixture as fx

pytest.importorskip("av")

HORIZON = 4
VIDEO_KEYS = {"observation.images.global": fx.CAM_D435, "observation.images.wrist": fx.CAM_ORBBEC}


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory):
    return fx.make_v3_dataset(tmp_path_factory.mktemp("lerobot_v3"))


def _dataset(root, **kwargs):
    return v3.LocalLeRobotV3Dataset(root, action_horizon=HORIZON, video_keys=VIDEO_KEYS, **kwargs)


def _gray(image: np.ndarray) -> float:
    return float(image.mean())


def test_sample_contents(dataset_root):
    dataset = _dataset(dataset_root)
    assert len(dataset) == sum(fx.EPISODE_LENGTHS)
    sample = dataset[fx.EPISODE_LENGTHS[0] + 2]  # episode 1, frame 2

    np.testing.assert_allclose(sample["observation.state"], np.full(7, 102.0))
    assert sample["action"].shape == (HORIZON, 7)
    np.testing.assert_allclose(sample["action"][:, 0], [102.5, 103.5, 104.5, 105.5])
    assert sample["prompt"] == fx.TASK_TEXT
    assert sample["task_index"] == 0
    for key in VIDEO_KEYS:
        assert sample[key].shape == (3, fx.HEIGHT, fx.WIDTH)
        assert sample[key].dtype == np.uint8
    assert abs(_gray(sample["observation.images.global"]) - fx.gray_level_d435(2)) < 6


def test_shared_video_file_uses_from_timestamp_offset(dataset_root):
    dataset = _dataset(dataset_root)
    for episode, frame in [(0, 0), (1, 0), (1, 3), (2, 4)]:
        index = dataset.indices_for(episode)[frame]
        offset = sum(fx.EPISODE_LENGTHS[:episode])
        image = dataset[index]["observation.images.wrist"]
        assert abs(_gray(image) - fx.gray_level_orbbec(offset + frame)) < 6, (episode, frame)


def test_action_padding_repeat_last(dataset_root):
    dataset = _dataset(dataset_root)
    last_of_episode_0 = dataset[fx.EPISODE_LENGTHS[0] - 1]["action"]
    np.testing.assert_allclose(last_of_episode_0[:, 0], np.full(HORIZON, 5.5))


def test_action_padding_zeros_with_hold_dims(dataset_root):
    dataset = _dataset(dataset_root, action_padding="zeros", action_padding_hold_dims=(6,))
    chunk = dataset[fx.EPISODE_LENGTHS[0] - 2]["action"]  # frames 4, 5, then two padded steps
    np.testing.assert_allclose(chunk[:2, 0], [4.5, 5.5])
    np.testing.assert_allclose(chunk[2:, :6], 0.0)  # velocity dims stop
    np.testing.assert_allclose(chunk[2:, 6], 5.5)  # the gripper target is held


def test_hold_dims_out_of_range_raises(dataset_root):
    with pytest.raises(ValueError, match="action_padding_hold_dims"):
        _dataset(dataset_root, action_padding="zeros", action_padding_hold_dims=(7,))


def test_episode_filters(dataset_root):
    assert _dataset(dataset_root, episode_indices=(0, 2)).episode_ids == [0, 2]
    assert _dataset(dataset_root, exclude_episode_indices=(1,)).episode_ids == [0, 2]
    assert _dataset(dataset_root, episode_indices=(0, 1, 2), exclude_episode_indices=(0,)).episode_ids == [1, 2]
    assert len(_dataset(dataset_root, episode_indices=(1,))) == fx.EPISODE_LENGTHS[1]
    with pytest.raises(ValueError, match="not found"):
        _dataset(dataset_root, episode_indices=(0, 7))
    with pytest.raises(ValueError, match="No episodes left"):
        _dataset(dataset_root, exclude_episode_indices=(0, 1, 2))


def test_indices_for(dataset_root):
    dataset = _dataset(dataset_root, episode_indices=(1, 2))
    assert [dataset[i]["observation.state"][0] for i in dataset.indices_for(1, stride=3)] == [100.0, 103.0, 106.0]
    # Only frames whose whole action chunk lies inside the episode: 8 - 4 + 1 = 5 frames for episode 1.
    assert len(dataset.indices_for(1, full_horizon_only=True)) == fx.EPISODE_LENGTHS[1] - HORIZON + 1
    assert dataset.episode_length(2) == fx.EPISODE_LENGTHS[2]


def test_unknown_video_key_raises(dataset_root):
    with pytest.raises(ValueError, match="Available"):
        v3.LocalLeRobotV3Dataset(
            dataset_root, action_horizon=HORIZON, video_keys={"observation.images.global": "observation.images.nope"}
        )


@pytest.mark.parametrize("task_column", ["task", "__index_level_0__"])
def test_task_text_from_either_column_layout(tmp_path, task_column):
    root = fx.make_v3_dataset(tmp_path, task_column=task_column)
    assert _dataset(root, episode_indices=(0,))[0]["prompt"] == fx.TASK_TEXT


def test_prompt_override(dataset_root):
    assert _dataset(dataset_root, prompt_override="put it in the bin")[3]["prompt"] == "put it in the bin"


def test_load_images_false_returns_placeholder(dataset_root):
    sample = _dataset(dataset_root, load_images=False)[0]
    assert sample["observation.images.global"].shape == (3, 1, 1)


def test_realman_configs_registered():
    action_only = _config.get_config("pi05_realman_action_only")
    full = _config.get_config("pi05_realman_full")
    overfit = _config.get_config("pi05_realman_full_overfit")

    for config in (action_only, full, overfit):
        assert config.model.pi05
        assert (config.model.action_dim, config.model.action_horizon) == (32, 16)
        assert config.data.repo_id == "local/realman_right_arm"
    assert (action_only.batch_size, full.batch_size, overfit.batch_size) == (8, 32, 8)
    assert action_only.ema_decay is None
    assert full.ema_decay == 0.999
    assert action_only.data.exclude_episode_indices == _config.REALMAN_HOLDOUT_EPISODES
    assert len(_config.REALMAN_HOLDOUT_EPISODES) == 13
    assert overfit.data.episode_indices == (0, 1)
    assert overfit.data.exclude_episode_indices == ()


def _smoke_config(root):
    config = _config.get_config("pi05_realman_action_only")
    data = dataclasses.replace(config.data, dataset_root=str(root), exclude_episode_indices=())
    return dataclasses.replace(config, data=data, batch_size=2, num_workers=0)


def test_one_batch_through_realman_config(dataset_root):
    """Loader -> repack -> RealmanInputs -> model transforms -> batch, with the real pi0.5 transform stack."""
    try:
        pi0_config.Pi0Config(pi05=True)
        from openpi.models import tokenizer as _tokenizer

        _tokenizer.PaligemmaTokenizer(200)
    except Exception as e:
        pytest.skip(f"PaliGemma tokenizer unavailable: {e}")

    config = _smoke_config(dataset_root)
    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=1)
    observation, actions = next(iter(loader))

    assert observation.state.shape == (2, 32)
    assert actions.shape == (2, 16, 32)
    assert set(observation.images) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert all(image.shape == (2, 224, 224, 3) for image in observation.images.values())
    assert bool(observation.image_masks["base_0_rgb"].all())
    assert not bool(observation.image_masks["right_wrist_0_rgb"].any())
    assert observation.tokenized_prompt.shape[0] == 2
    # Dimensions 7..31 are zero padding.
    np.testing.assert_allclose(np.asarray(actions)[..., 7:], 0.0)


@pytest.mark.parametrize("name", ["pi05_realman_action_only", "pi05_realman_full", "pi05_realman_full_overfit", "pi05_realman_multitask_action_only"])
def test_realman_configs_parse_on_the_command_line(name, monkeypatch):
    """scripts/train.py builds its CLI from the registered configs; tuple defaults must be understood by tyro."""
    monkeypatch.setattr("sys.argv", ["train.py", name, "--exp-name", "x"])

    config = _config.cli()

    assert config.name == name
    assert config.data.exclude_episode_indices == _config.get_config(name).data.exclude_episode_indices
