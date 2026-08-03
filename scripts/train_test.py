import dataclasses
import os
import pathlib

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config as _config

from . import train


def test_truncate_local_metrics(tmp_path: pathlib.Path):
    metrics_path = tmp_path / "metrics.csv"
    for step in (0, 100, 200, 300):
        train._write_local_metrics(  # noqa: SLF001
            metrics_path,
            step,
            {"loss": 1.0, "grad_norm": 2.0, "param_norm": 3.0, "learning_rate": 4.0},
        )

    train._truncate_local_metrics(metrics_path, 201)  # noqa: SLF001

    assert [line.split(",", 1)[0] for line in metrics_path.read_text().splitlines()] == ["step", "0", "100", "200"]


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)
