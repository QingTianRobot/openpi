"""Builds a tiny synthetic LeRobot v3 dataset for tests (real parquet and mp4 files, a few frames each).

Layout mirrors what the RealMan dataset view looks like:
  * one parquet file per episode with `observation.state` (7), `action` (7) and `task_index`;
  * camera `observation.images.d435`: one mp4 per episode;
  * camera `observation.images.orbbec-right`: ONE mp4 shared by all episodes, addressed with `from_timestamp`.

State and action values encode where they come from: `state[t] = 100 * episode + t`, `action[t] = state[t] + 0.5`.
Video frames are flat gray images whose level encodes the frame, see `gray_level_d435` / `gray_level_orbbec`.
"""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FPS = 30
EPISODE_LENGTHS = (6, 8, 5)
HEIGHT, WIDTH = 48, 64
CAM_D435 = "observation.images.d435"
CAM_ORBBEC = "observation.images.orbbec-right"
TASK_TEXT = "pick the pen"


def gray_level_d435(frame_in_episode: int) -> int:
    return 10 + 10 * frame_in_episode


def gray_level_orbbec(frame_in_file: int) -> int:
    """The shared file holds all episodes back to back, so the level follows the global frame number."""
    return 10 + 10 * frame_in_file


def write_video(path: Path, levels: list[int]) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        stream.options = {"crf": "10", "g": "1", "preset": "ultrafast"}
        for level in levels:
            image = np.full((HEIGHT, WIDTH, 3), level, dtype=np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")))
        container.mux(stream.encode())


def _list_array(values: np.ndarray) -> pa.Array:
    return pa.array(values.tolist(), type=pa.list_(pa.float32()))


def make_v3_dataset(root: Path, *, task_column: str = "task") -> Path:
    """Write the dataset under `root`. `task_column` is `task` or `__index_level_0__` (pandas-written files)."""
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    info = {"codebase_version": "v3.0", "fps": FPS, "total_episodes": len(EPISODE_LENGTHS)}
    info["total_frames"] = sum(EPISODE_LENGTHS)
    (root / "meta" / "info.json").write_text(json.dumps(info))
    pq.write_table(pa.table({"task_index": [0], task_column: [TASK_TEXT]}), root / "meta" / "tasks.parquet")

    meta_rows = []
    start = 0
    orbbec_levels: list[int] = []
    for episode, length in enumerate(EPISODE_LENGTHS):
        frames = np.arange(length)
        state = np.repeat((100 * episode + frames)[:, None], 7, axis=1).astype(np.float32)
        table = pa.table(
            {
                "observation.state": _list_array(state),
                "action": _list_array(state + 0.5),
                "task_index": pa.array(np.zeros(length, dtype=np.int64)),
                "episode_index": pa.array(np.full(length, episode, dtype=np.int64)),
                "frame_index": pa.array(frames.astype(np.int64)),
                "index": pa.array(start + frames.astype(np.int64)),
            }
        )
        data_dir = root / "data" / "chunk-000"
        data_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, data_dir / f"file-{episode:03d}.parquet")
        write_video(
            root / "videos" / CAM_D435 / "chunk-000" / f"file-{episode:03d}.mp4",
            [gray_level_d435(t) for t in range(length)],
        )
        orbbec_offset = len(orbbec_levels)
        orbbec_levels += [gray_level_orbbec(orbbec_offset + t) for t in range(length)]
        meta_rows.append(
            {
                "episode_index": episode,
                "length": length,
                "data/chunk_index": 0,
                "data/file_index": episode,
                "dataset_from_index": start,
                "dataset_to_index": start + length,
                f"videos/{CAM_D435}/chunk_index": 0,
                f"videos/{CAM_D435}/file_index": episode,
                f"videos/{CAM_D435}/from_timestamp": 0.0,
                f"videos/{CAM_D435}/to_timestamp": length / FPS,
                f"videos/{CAM_ORBBEC}/chunk_index": 0,
                f"videos/{CAM_ORBBEC}/file_index": 0,
                f"videos/{CAM_ORBBEC}/from_timestamp": orbbec_offset / FPS,
                f"videos/{CAM_ORBBEC}/to_timestamp": (orbbec_offset + length) / FPS,
            }
        )
        start += length
    write_video(root / "videos" / CAM_ORBBEC / "chunk-000" / "file-000.mp4", orbbec_levels)
    pq.write_table(pa.Table.from_pylist(meta_rows), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    return root
