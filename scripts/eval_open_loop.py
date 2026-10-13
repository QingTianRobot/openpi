"""Open-loop evaluation of a trained policy on held-out episodes of a local LeRobot v3 dataset.

For frames sampled from each held-out episode the policy is asked for an action chunk and compared with the
recorded one. Nothing is executed on a robot. The script reports the mean absolute error per action dimension
and per chunk step, next to two reference predictors (all zeros, and the mean recorded action), so the numbers
can be judged against "what a trivial model would score".

Example:
    uv run scripts/eval_open_loop.py --config-name pi05_realman_action_only \\
        --checkpoint-dir checkpoints/pi05_realman_action_only/<exp>/<step>
By default the evaluated episodes are the config's `exclude_episode_indices` (the held-out set).
"""

from collections.abc import Callable, Sequence
import dataclasses
import json
import logging
import pathlib
from typing import Any

import numpy as np
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.training import lerobot_v3_dataset as _v3


def make_repack(data_config: _config.DataConfig) -> Callable[[dict], dict]:
    """Chain the config's repack transforms: dataset item -> the dict the policy server would receive."""

    def repack(item: dict) -> dict:
        for transform in data_config.repack_transforms.inputs:
            item = transform(item)
        return item

    return repack


def summarize(pred: np.ndarray, truth: np.ndarray, episodes: np.ndarray) -> dict[str, Any]:
    """Error statistics for predictions and recorded actions of shape (samples, horizon, dim)."""
    if pred.shape != truth.shape or pred.ndim != 3:
        raise ValueError(f"pred and truth must both be (samples, horizon, dim), got {pred.shape} and {truth.shape}")
    error = np.abs(pred - truth)
    zero_error = np.abs(truth)
    mean_error = np.abs(truth - truth.mean(axis=(0, 1), keepdims=True))
    per_dim = error.mean(axis=(0, 1))
    return {
        "num_samples": int(pred.shape[0]),
        "num_episodes": len(np.unique(episodes)),
        "mae": float(error.mean()),
        "mae_per_dim": per_dim.tolist(),
        "mae_per_step": error.mean(axis=(0, 2)).tolist(),
        "mae_per_episode": {int(e): float(error[episodes == e].mean()) for e in np.unique(episodes)},
        "baseline_zeros": {"mae": float(zero_error.mean()), "mae_per_dim": zero_error.mean(axis=(0, 1)).tolist()},
        "baseline_mean_action": {"mae": float(mean_error.mean()), "mae_per_dim": mean_error.mean(axis=(0, 1)).tolist()},
        "relative_to_zeros_per_dim": (per_dim / np.maximum(zero_error.mean(axis=(0, 1)), 1e-9)).tolist(),
    }


def evaluate(
    policy,
    dataset,
    repack: Callable[[dict], dict],
    *,
    stride: int = 10,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Run `policy.infer` on frames 0, stride, 2*stride... of every episode whose action chunk lies fully inside it."""
    preds, truths, owners = [], [], []
    for episode in dataset.episode_ids:
        for index in dataset.indices_for(episode, stride, full_horizon_only=True):
            observation = repack(dataset[index])
            truth = np.asarray(observation.pop("actions"), dtype=np.float32)
            pred = np.asarray(policy.infer(observation)["actions"], dtype=np.float32)
            if pred.shape != truth.shape:
                raise ValueError(f"policy returned {pred.shape}, the dataset action chunk is {truth.shape}")
            preds.append(pred)
            truths.append(truth)
            owners.append(episode)
            if max_samples is not None and len(preds) >= max_samples:
                return summarize(np.stack(preds), np.stack(truths), np.asarray(owners))
    if not preds:
        raise ValueError("No samples to evaluate: every episode is shorter than the action horizon")
    return summarize(np.stack(preds), np.stack(truths), np.asarray(owners))


def format_report(result: dict[str, Any]) -> str:
    dims = [f"d{i}" for i in range(len(result["mae_per_dim"]))]
    rows = [
        ("policy", result["mae_per_dim"], result["mae"]),
        ("zeros", result["baseline_zeros"]["mae_per_dim"], result["baseline_zeros"]["mae"]),
        ("mean action", result["baseline_mean_action"]["mae_per_dim"], result["baseline_mean_action"]["mae"]),
    ]
    lines = [
        f"{result['num_samples']} samples from {result['num_episodes']} episodes",
        f"{'':<12}" + "".join(f"{d:>9}" for d in dims) + f"{'all':>9}",
    ]
    lines += [
        f"{name:<12}" + "".join(f"{v:>9.4f}" for v in per_dim) + f"{total:>9.4f}" for name, per_dim, total in rows
    ]
    return "\n".join(lines)


@dataclasses.dataclass
class Args:
    config_name: str
    checkpoint_dir: str
    # Episodes to evaluate. Empty means the config's held-out episodes.
    episodes: Sequence[int] = ()
    # Evaluate every `stride`-th frame of each episode.
    stride: int = 10
    max_samples: int | None = None
    # Write the full result as JSON.
    output: pathlib.Path | None = None


def main(args: Args) -> None:
    train_config = _config.get_config(args.config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    episodes = list(args.episodes) or list(data_config.exclude_episode_indices)
    if not episodes:
        raise SystemExit("No episodes to evaluate: pass --episodes or use a config with exclude_episode_indices")

    dataset = _v3.LocalLeRobotV3Dataset(
        data_config.dataset_root,
        action_horizon=train_config.model.action_horizon,
        video_keys=data_config.video_keys,
        episode_indices=episodes,
        action_padding=data_config.action_padding,
        action_padding_hold_dims=data_config.action_padding_hold_dims,
        prompt_override=data_config.prompt_override,
    )
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint_dir)
    result = evaluate(policy, dataset, make_repack(data_config), stride=args.stride, max_samples=args.max_samples)
    print(format_report(result))
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2))
        logging.info("Wrote %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
