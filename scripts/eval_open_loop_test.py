import dataclasses

import numpy as np
import pytest

from openpi.training import config as _config
from openpi.training import lerobot_v3_dataset as v3
from openpi.training import lerobot_v3_fixture as fx
from scripts import eval_open_loop

pytest.importorskip("av")

HORIZON = 4


class _Oracle:
    """Returns the recorded actions, found through the state (state value + 0.5 == first action)."""

    def infer(self, observation):
        first = float(observation["state"][0]) + 0.5
        return {"actions": np.stack([np.full(7, first + step, dtype=np.float32) for step in range(HORIZON)])}


class _Zeros:
    def infer(self, observation):
        return {"actions": np.zeros((HORIZON, 7), dtype=np.float32)}


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = fx.make_v3_dataset(tmp_path_factory.mktemp("eval_v3"))
    config = _config.get_config("pi05_realman_action_only")
    data_config = dataclasses.replace(config.data, dataset_root=str(root), exclude_episode_indices=()).create(
        config.assets_dirs, config.model
    )
    dataset = v3.LocalLeRobotV3Dataset(
        root,
        action_horizon=HORIZON,
        video_keys=data_config.video_keys,
        episode_indices=(0, 1, 2),
    )
    return dataset, eval_open_loop.make_repack(data_config)


def test_oracle_policy_has_zero_error(setup):
    dataset, repack = setup

    result = eval_open_loop.evaluate(_Oracle(), dataset, repack, stride=1)

    # Frames with a full chunk: (6-4+1) + (8-4+1) + (5-4+1) = 3 + 5 + 2.
    assert result["num_samples"] == 10
    assert result["num_episodes"] == 3
    assert result["mae"] == pytest.approx(0.0, abs=1e-5)
    assert len(result["mae_per_dim"]) == 7
    assert len(result["mae_per_step"]) == HORIZON


def test_zero_policy_matches_zero_baseline(setup):
    dataset, repack = setup

    result = eval_open_loop.evaluate(_Zeros(), dataset, repack, stride=1)

    assert result["mae"] == pytest.approx(result["baseline_zeros"]["mae"])
    assert result["relative_to_zeros_per_dim"] == pytest.approx([1.0] * 7)
    assert result["baseline_mean_action"]["mae"] < result["baseline_zeros"]["mae"]


def test_stride_and_max_samples(setup):
    dataset, repack = setup

    assert eval_open_loop.evaluate(_Oracle(), dataset, repack, stride=2)["num_samples"] == 2 + 3 + 1
    assert eval_open_loop.evaluate(_Oracle(), dataset, repack, stride=1, max_samples=4)["num_samples"] == 4


def test_shape_mismatch_and_empty_inputs_raise(setup):
    dataset, repack = setup

    class _Wrong:
        def infer(self, observation):
            return {"actions": np.zeros((HORIZON + 1, 7), dtype=np.float32)}

    with pytest.raises(ValueError, match="policy returned"):
        eval_open_loop.evaluate(_Wrong(), dataset, repack)
    with pytest.raises(ValueError, match="must both be"):
        eval_open_loop.summarize(np.zeros((2, 4, 7)), np.zeros((2, 4, 6)), np.zeros(2))


def test_report_lists_policy_and_baselines(setup):
    dataset, repack = setup
    report = eval_open_loop.format_report(eval_open_loop.evaluate(_Oracle(), dataset, repack, stride=1))

    assert "policy" in report
    assert "zeros" in report
    assert "mean action" in report
    assert "10 samples from 3 episodes" in report
