import openpi.training.config as _config


def test_dataset_root_defaults_to_the_container_mount(monkeypatch):
    monkeypatch.delenv("OPENPI_REALMAN_DATASET_DIR", raising=False)
    assert _config.LeRobotRealmanDataConfig().dataset_root == "/datasets/lerobot_realman"


def test_dataset_root_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENPI_REALMAN_DATASET_DIR", str(tmp_path))
    assert _config.LeRobotRealmanDataConfig().dataset_root == str(tmp_path)


def test_smoke_config_holds_out_one_episode_and_has_its_own_assets():
    cfg = _config.get_config("pi05_realman_smoke")
    assert cfg.data.exclude_episode_indices == (2,)
    assert cfg.data.repo_id == "local/realman_smoke"
    assert cfg.num_train_steps == 20
