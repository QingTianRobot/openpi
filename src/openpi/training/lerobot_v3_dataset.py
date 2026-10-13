from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import json
from pathlib import Path
from typing import Literal, SupportsIndex

import numpy as np
import pyarrow.parquet as pq
from torchcodec.decoders import VideoDecoder

# Output camera key -> video key in the dataset. The identity mapping is what the Dobot recorder produces.
DEFAULT_VIDEO_KEYS: Mapping[str, str] = {
    "observation.images.global": "observation.images.global",
    "observation.images.wrist": "observation.images.wrist",
}

ActionPadding = Literal["repeat_last", "zeros"]


@dataclasses.dataclass(frozen=True)
class _Episode:
    episode_index: int
    states: np.ndarray  # (T, state_dim) float32
    actions: np.ndarray  # (T, action_dim) float32
    task_indices: np.ndarray  # (T,) int64
    # Output camera key -> (video key, chunk index, file index, first frame of this episode inside that file).
    videos: dict[str, tuple[str, int, int, int]]

    def __len__(self) -> int:
        return len(self.states)


def _read_tasks(root: Path) -> dict[int, str]:
    """Read `meta/tasks.parquet`. The task text is the `task` column or, for pandas-written files, the index."""
    table = pq.read_table(root / "meta" / "tasks.parquet")
    names = table.schema.names
    if "task_index" not in names:
        raise ValueError(f"tasks.parquet has no `task_index` column: {names}")
    text_column = next((n for n in ("task", "__index_level_0__") if n in names), None)
    if text_column is None:
        raise ValueError(f"tasks.parquet has no task text column (`task` or the pandas index): {names}")
    return {
        int(i): str(t) for i, t in zip(table["task_index"].to_pylist(), table[text_column].to_pylist(), strict=True)
    }


def table_action_dim(table, action_key: str) -> int:
    return len(table[action_key][0].as_py())


def _as_float_matrix(column, name: str, expected_rows: int) -> np.ndarray:
    values = np.asarray(column.to_pylist(), dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != expected_rows:
        raise ValueError(f"column `{name}` must be a (rows, dim) matrix with {expected_rows} rows, got {values.shape}")
    return values


class LocalLeRobotV3Dataset:
    """Minimal read-only adapter for local LeRobot v3 Parquet/MP4 datasets.

    Each item holds one frame: the state, one decoded image per output camera key, the next `action_horizon`
    actions and the task text. The dataset must already contain `observation.state` and `action` columns, which
    is what the RealMan and Dobot dataset views provide.

    Args:
        root: Dataset root directory (contains `meta/`, `data/`, `videos/`).
        action_horizon: Number of consecutive actions returned per item.
        load_images: If false, images are replaced by a 1x1 placeholder so statistics can be computed quickly.
        video_keys: Output camera key -> video key in the dataset (see `DEFAULT_VIDEO_KEYS`).
        episode_indices: If given, only these episodes are used. Every listed episode must exist.
        exclude_episode_indices: Episodes to leave out, e.g. a held-out validation set.
        action_padding: How to fill action chunks that run past the end of an episode. `repeat_last` repeats the
            final action. `zeros` appends zeros, which for velocity commands means "stop".
        action_padding_hold_dims: Action dimensions that keep repeating the final value even with `zeros` padding,
            e.g. a gripper target, where zero would mean "close" rather than "do nothing".
        prompt_override: If set, replaces the task text of every item (the prompt used at inference must match).
        state_key: Parquet column holding the state vector.
        action_key: Parquet column holding the action vector.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        action_horizon: int,
        load_images: bool = True,
        video_keys: Mapping[str, str] | None = None,
        episode_indices: Sequence[int] | None = None,
        exclude_episode_indices: Sequence[int] = (),
        action_padding: ActionPadding = "repeat_last",
        action_padding_hold_dims: Sequence[int] = (),
        prompt_override: str | None = None,
        state_key: str = "observation.state",
        action_key: str = "action",
    ):
        if action_padding not in ("repeat_last", "zeros"):
            raise ValueError(f"action_padding must be 'repeat_last' or 'zeros', got {action_padding!r}")
        self.root = Path(root)
        self.action_horizon = action_horizon
        self.load_images = load_images
        self.action_padding = action_padding
        self.action_padding_hold_dims = tuple(action_padding_hold_dims)
        self.prompt_override = prompt_override
        self.video_keys = dict(DEFAULT_VIDEO_KEYS if video_keys is None else video_keys)
        self._decoders: dict[tuple[str, int, int], VideoDecoder] = {}

        with (self.root / "meta" / "info.json").open() as f:
            info = json.load(f)
        if info.get("codebase_version") != "v3.0":
            raise ValueError(f"Expected LeRobot v3.0 dataset, got {info.get('codebase_version')!r}")
        self.fps = float(info["fps"])
        self.tasks = _read_tasks(self.root)

        episode_files = sorted((self.root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        if not episode_files:
            raise ValueError(f"No episode metadata found under {self.root / 'meta' / 'episodes'}")
        available_videos = sorted(
            {
                n.split("/")[1]
                for n in pq.read_schema(episode_files[0]).names
                if n.startswith("videos/") and n.endswith("/chunk_index")
            }
        )
        missing = sorted(set(self.video_keys.values()) - set(available_videos))
        if missing:
            raise ValueError(f"Video keys {missing} not found in the dataset. Available: {available_videos}")

        def video_columns(video_key: str) -> list[str]:
            base = f"videos/{video_key}/"
            names = pq.read_schema(episode_files[0]).names
            return [base + "chunk_index", base + "file_index"] + (
                [base + "from_timestamp"] if base + "from_timestamp" in names else []
            )

        columns = ["episode_index", "length", "data/chunk_index", "data/file_index"]
        for video_key in sorted(set(self.video_keys.values())):
            columns += video_columns(video_key)
        episode_meta = []
        for path in episode_files:
            episode_meta.extend(pq.read_table(path, columns=columns).to_pylist())
        episode_meta.sort(key=lambda row: int(row["episode_index"]))

        present = {int(row["episode_index"]) for row in episode_meta}
        if episode_indices is not None:
            unknown = sorted(set(episode_indices) - present)
            if unknown:
                raise ValueError(f"episode_indices {unknown} not found in the dataset (it has {len(present)} episodes)")
        wanted = present if episode_indices is None else set(episode_indices)
        wanted -= set(exclude_episode_indices)
        if not wanted:
            raise ValueError("No episodes left after applying episode_indices / exclude_episode_indices")

        self.episodes: list[_Episode] = []
        self._samples: list[tuple[int, int]] = []
        self._start: dict[int, int] = {}
        for meta in episode_meta:
            episode_index = int(meta["episode_index"])
            if episode_index not in wanted:
                continue
            length = int(meta["length"])
            data_path = (
                self.root
                / "data"
                / f"chunk-{int(meta['data/chunk_index']):03d}"
                / f"file-{int(meta['data/file_index']):03d}.parquet"
            )
            table = pq.read_table(data_path, columns=[state_key, action_key, "task_index"])
            if table.num_rows != length:
                raise ValueError(f"{data_path} has {table.num_rows} rows but the episode metadata says {length}")
            videos = {}
            for out_key, video_key in self.video_keys.items():
                base = f"videos/{video_key}/"
                offset = round(float(meta.get(base + "from_timestamp") or 0.0) * self.fps)
                videos[out_key] = (video_key, int(meta[base + "chunk_index"]), int(meta[base + "file_index"]), offset)
            if any(not 0 <= d < table_action_dim(table, action_key) for d in self.action_padding_hold_dims):
                raise ValueError(
                    f"action_padding_hold_dims {self.action_padding_hold_dims} out of range for `{action_key}`"
                )
            episode = _Episode(
                episode_index=episode_index,
                states=_as_float_matrix(table[state_key], state_key, length),
                actions=_as_float_matrix(table[action_key], action_key, length),
                task_indices=np.asarray(table["task_index"].to_pylist(), dtype=np.int64),
                videos=videos,
            )
            position = len(self.episodes)
            self._start[episode_index] = len(self._samples)
            self.episodes.append(episode)
            self._samples.extend((position, frame) for frame in range(length))
        self._position_of = {ep.episode_index: i for i, ep in enumerate(self.episodes)}

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def episode_ids(self) -> list[int]:
        return [ep.episode_index for ep in self.episodes]

    def episode_length(self, episode_index: int) -> int:
        return len(self.episodes[self._position_of[episode_index]])

    def indices_for(self, episode_index: int, stride: int = 1, *, full_horizon_only: bool = False) -> list[int]:
        """Dataset indices of frames 0, stride, 2*stride, ... of an episode.

        With `full_horizon_only`, frames whose action chunk would run past the episode end are skipped.
        """
        position = self._position_of[episode_index]
        length = len(self.episodes[position])
        last = length - self.action_horizon if full_horizon_only else length - 1
        start = self._start[episode_index]
        return [start + frame for frame in range(0, last + 1, stride)]

    def _decoder(self, video_key: str, chunk: int, file_index: int) -> VideoDecoder:
        key = (video_key, chunk, file_index)
        if key not in self._decoders:
            path = self.root / "videos" / video_key / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
            self._decoders[key] = VideoDecoder(str(path))
        return self._decoders[key]

    def _frame(self, out_key: str, episode: _Episode, frame_index: int) -> np.ndarray:
        if not self.load_images:
            return np.zeros((3, 1, 1), dtype=np.uint8)
        video_key, chunk, file_index, offset = episode.videos[out_key]
        return self._decoder(video_key, chunk, file_index)[offset + frame_index].numpy()

    def _action_chunk(self, episode: _Episode, frame_index: int) -> np.ndarray:
        steps = frame_index + np.arange(self.action_horizon)
        last = len(episode) - 1
        chunk = episode.actions[np.minimum(steps, last)]
        if self.action_padding == "zeros":
            padded = np.zeros_like(chunk)
            hold = list(self.action_padding_hold_dims)
            padded[:, hold] = chunk[:, hold]
            chunk = np.where((steps > last)[:, None], padded, chunk)
        return chunk

    def __getitem__(self, index: SupportsIndex) -> dict:
        episode_position, frame_index = self._samples[int(index)]
        episode = self.episodes[episode_position]
        task_index = int(episode.task_indices[frame_index])
        return {
            "observation.state": episode.states[frame_index],
            **{out_key: self._frame(out_key, episode, frame_index) for out_key in self.video_keys},
            "action": self._action_chunk(episode, frame_index),
            "task_index": task_index,
            "prompt": self.prompt_override if self.prompt_override is not None else self.tasks[task_index],
        }
