import dataclasses

import pytest

from openpi.shared import normalize
from openpi.training import config as _config
from openpi.training import lerobot_v3_fixture as fx
from scripts import compute_norm_stats

pytest.importorskip("av")


def test_norm_stats_for_realman_config(tmp_path, monkeypatch):
    root = fx.make_v3_dataset(tmp_path / "dataset")
    config = _config.get_config("pi05_realman_action_only")
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(config.data, dataset_root=str(root), exclude_episode_indices=()),
        batch_size=2,
        num_workers=0,
        assets_base_dir=str(tmp_path / "assets"),
    )
    monkeypatch.setattr(_config, "get_config", lambda name: config)

    compute_norm_stats.main("pi05_realman_action_only")

    stats = normalize.load(config.assets_dirs / "local/realman_right_arm")
    assert set(stats) == {"state", "actions"}
    assert stats["state"].mean.shape == (7,)
    assert stats["actions"].q01.shape == (7,)
    # state[t] = 100 * episode + t over 19 frames; 9 batches of 2 frames are consumed (the odd frame is dropped).
    assert 0 < float(stats["state"].mean[0]) < 210
    assert float(stats["state"].q99[0]) > float(stats["state"].q01[0])
