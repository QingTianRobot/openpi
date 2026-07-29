from pathlib import Path

import pytest

from scripts.docker.workflow_test import _run_host_function


@pytest.fixture
def repo() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("argument", "expected_uri", "expected_force"),
    [
        ("pi0_base", "gs://openpi-assets/checkpoints/pi0_base", "false"),
        ("gs://custom-bucket/checkpoints/model --force", "gs://custom-bucket/checkpoints/model", "true"),
    ],
)
def test_host_download_runs_in_disposable_cpu_container(
    repo: Path,
    tmp_path: Path,
    argument: str,
    expected_uri: str,
    expected_force: str,
) -> None:
    result, calls, _ = _run_host_function(repo, tmp_path, f"opid-download {argument}")

    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    call = calls[0][1:]
    assert call[:12] == [
        "compose",
        "-f",
        str(repo / "compose.yaml"),
        "run",
        "--rm",
        "-T",
        "openpi",
        "uv",
        "run",
        "--no-sync",
        "python",
        "-c",
    ]
    assert "download.maybe_download" in call[12]
    assert call[13:] == [expected_uri, expected_force]


@pytest.mark.parametrize("invocation", ["opid-download", "opid-download pi0_base --invalid"])
def test_host_download_rejects_invalid_arguments_before_docker(repo: Path, tmp_path: Path, invocation: str) -> None:
    result, calls, _ = _run_host_function(repo, tmp_path, invocation)

    assert result.returncode == 2
    assert "Usage: opid-download" in result.stderr
    assert calls == []
