from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import SupportsIndex

import numpy as np
import pyarrow.parquet as pq
from torchcodec.decoders import VideoDecoder


@dataclasses.dataclass(frozen=True)
class _Episode:
    episode_index: int
    rows: list[dict]
    video_chunk: int
    video_file: int


class LocalLeRobotV3Dataset:
    """Minimal read-only adapter for local LeRobot v3 Parquet/MP4 datasets."""

    def __init__(self, root: str | Path, *, action_horizon: int, load_images: bool = True):
        self.root = Path(root)
        self.action_horizon = action_horizon
        self.load_images = load_images
        self._decoders: dict[tuple[str, int, int], VideoDecoder] = {}

        with (self.root / "meta" / "info.json").open() as f:
            info = json.load(f)
        if info.get("codebase_version") != "v3.0":
            raise ValueError(f"Expected LeRobot v3.0 dataset, got {info.get('codebase_version')!r}")

        tasks_table = pq.read_table(self.root / "meta" / "tasks.parquet")
        self.tasks = {
            int(row["task_index"]): str(row["task"])
            for row in tasks_table.to_pylist()
        }

        episode_files = sorted((self.root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        episode_meta = []
        for path in episode_files:
            episode_meta.extend(pq.read_table(path).to_pylist())
        episode_meta.sort(key=lambda row: int(row["episode_index"]))

        self.episodes: list[_Episode] = []
        self._samples: list[tuple[int, int]] = []
        for meta in episode_meta:
            data_path = self.root / "data" / f"chunk-{int(meta['data/chunk_index']):03d}" / (
                f"file-{int(meta['data/file_index']):03d}.parquet"
            )
            rows = pq.read_table(data_path).to_pylist()
            episode = _Episode(
                episode_index=int(meta["episode_index"]),
                rows=rows,
                video_chunk=int(meta["videos/observation.images.global/chunk_index"]),
                video_file=int(meta["videos/observation.images.global/file_index"]),
            )
            episode_position = len(self.episodes)
            self.episodes.append(episode)
            self._samples.extend((episode_position, frame) for frame in range(len(rows)))

    def __len__(self) -> int:
        return len(self._samples)

    def _decoder(self, camera: str, chunk: int, file_index: int) -> VideoDecoder:
        key = (camera, chunk, file_index)
        if key not in self._decoders:
            path = self.root / "videos" / camera / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
            self._decoders[key] = VideoDecoder(str(path))
        return self._decoders[key]

    def _frame(self, camera: str, episode: _Episode, frame_index: int) -> np.ndarray:
        if not self.load_images:
            return np.zeros((3, 1, 1), dtype=np.uint8)
        return self._decoder(camera, episode.video_chunk, episode.video_file)[frame_index].numpy()

    def __getitem__(self, index: SupportsIndex) -> dict:
        episode_position, frame_index = self._samples[int(index)]
        episode = self.episodes[episode_position]
        row = episode.rows[frame_index]
        final_index = len(episode.rows) - 1
        actions = np.asarray(
            [
                episode.rows[min(frame_index + offset, final_index)]["action"]
                for offset in range(self.action_horizon)
            ],
            dtype=np.float32,
        )
        task_index = int(row["task_index"])
        return {
            "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
            "observation.images.global": self._frame(
                "observation.images.global", episode, frame_index
            ),
            "observation.images.wrist": self._frame(
                "observation.images.wrist", episode, frame_index
            ),
            "action": actions,
            "task_index": task_index,
            "prompt": self.tasks[task_index],
        }
