import openpi.training.config as _config


def test_dataset_root_defaults_to_the_container_mount(monkeypatch):
    monkeypatch.delenv("OPENPI_REALMAN_DATASET_DIR", raising=False)
    assert _config.LeRobotRealmanDataConfig().dataset_root == "/datasets/lerobot_realman"


def test_dataset_root_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENPI_REALMAN_DATASET_DIR", str(tmp_path))
    assert _config.LeRobotRealmanDataConfig().dataset_root == str(tmp_path)


def test_wrist_camera_defaults_to_the_right_arm():
    assert _config.realman_video_keys() == _config.REALMAN_VIDEO_KEYS
    assert _config.REALMAN_VIDEO_KEYS["observation.images.wrist"] == "observation.images.orbbec-right"


def test_sjd_left_config_reads_the_left_wrist_camera():
    train = _config.get_config("pi05_realman_sjd_left_v1_action_only")
    assert _config.realman_video_keys(train.data.wrist_camera) == {
        "observation.images.global": "observation.images.d435",
        "observation.images.wrist": "observation.images.orbbec-left",
    }
    assert train.data.exclude_episode_indices == tuple(range(9, 87, 10))
    assert train.keep_period is None
    assert train.batch_size == 16
