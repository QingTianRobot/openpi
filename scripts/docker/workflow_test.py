import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def repo() -> Path:
    return Path(__file__).resolve().parents[2]


def _render_service(repo: Path, *compose_files: str) -> dict:
    data_dir = Path("/tmp/openpi-compose-test-data")
    env = {
        **os.environ,
        "OPENPI_COMPOSE_PROJECT_NAME": "openpi-workflow-test",
        "OPENPI_DOCKER_DATA_DIR": str(data_dir),
        "OPENPI_BUILD_ALL_PROXY": "socks5://host.docker.internal:17891",
        "OPENPI_BUILD_HTTP_PROXY": "http://host.docker.internal:17891",
        "OPENPI_BUILD_HTTPS_PROXY": "http://host.docker.internal:17891",
        "OPENPI_BUILD_NO_PROXY": "localhost,127.0.0.1",
        "OPENPI_GPU_DEVICE": "2",
        "OPENPI_GID": "2345",
        "OPENPI_UBUNTU_APT_MIRROR": "https://mirror.example/ubuntu",
        "OPENPI_UBUNTU_SECURITY_MIRROR": "https://security-mirror.example/ubuntu",
        "OPENPI_UID": "1234",
        "OPENPI_UV_INDEX_URL": "https://python-mirror.example/simple",
        "OPENPI_UV_PYTHON_ATTEMPTS": "4",
        "OPENPI_UV_PYTHON_INSTALL_MIRROR": "https://python-download.example/releases/download",
        "OPENPI_UV_HTTP_TIMEOUT": "420",
    }
    command = ["docker", "compose"]
    for compose_file in compose_files:
        command.extend(["-f", str(repo / compose_file)])
    command.extend(["config", "--format", "json"])
    result = subprocess.run(command, cwd=repo, env=env, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)["services"]["openpi"]


def _volumes_by_target(service: dict) -> dict[str, dict]:
    return {volume["target"]: volume for volume in service["volumes"]}


def test_base_compose_is_a_non_root_persistent_dev_service(repo: Path) -> None:
    service = _render_service(repo, "compose.yaml")
    volumes = _volumes_by_target(service)

    assert service["user"] == "1234:2345"
    assert service["working_dir"] == "/workspace"
    assert service["command"] == ["sleep", "infinity"]
    assert "gpus" not in service
    assert Path(volumes["/workspace"]["source"]) == repo
    assert volumes["/workspace"]["type"] == "bind"

    data_dir = Path("/tmp/openpi-compose-test-data")
    expected_persistent_binds = {
        "/home/openpi/.cache": data_dir / "cache",
        "/workspace/assets": data_dir / "assets",
        "/workspace/checkpoints": data_dir / "checkpoints",
        "/workspace/data": data_dir / "data",
        "/workspace/wandb": data_dir / "wandb",
    }
    for target, source in expected_persistent_binds.items():
        assert volumes[target]["type"] == "bind"
        assert Path(volumes[target]["source"]) == source

    assert service["environment"]["OPENPI_DATA_HOME"] == "/home/openpi/.cache/openpi"
    assert service["environment"]["UV_PROJECT_ENVIRONMENT"] == "/.venv"
    assert service["build"]["args"]["UBUNTU_APT_MIRROR"] == "https://mirror.example/ubuntu"
    assert service["build"]["args"]["UBUNTU_SECURITY_MIRROR"] == "https://security-mirror.example/ubuntu"
    assert service["build"]["args"]["UV_INDEX_URL"] == "https://python-mirror.example/simple"
    assert service["build"]["args"]["UV_HTTP_TIMEOUT"] == "420"
    assert service["build"]["args"]["UV_PYTHON_INSTALL_MIRROR"] == (
        "https://python-download.example/releases/download"
    )
    assert service["build"]["args"]["OPENPI_UV_PYTHON_ATTEMPTS"] == "4"


def test_gpu_compose_selects_only_the_requested_gpu(repo: Path) -> None:
    service = _render_service(repo, "compose.yaml", "compose.gpu.yaml")

    assert service["gpus"] == [{"driver": "nvidia", "device_ids": ["2"]}]
    assert service["environment"]["NVIDIA_DRIVER_CAPABILITIES"] == "compute,utility"
    assert service["shm_size"] == "2147483648"


def test_proxy_compose_passes_build_only_proxy_and_host_gateway(repo: Path) -> None:
    service = _render_service(repo, "compose.yaml", "compose.proxy.yaml")
    build = service["build"]

    assert build["extra_hosts"] == ["host.docker.internal=host-gateway"]
    assert build["args"]["HTTP_PROXY"] == "http://host.docker.internal:17891"
    assert build["args"]["HTTPS_PROXY"] == "http://host.docker.internal:17891"
    assert build["args"]["ALL_PROXY"] == "socks5://host.docker.internal:17891"
    assert build["args"]["NO_PROXY"] == "localhost,127.0.0.1"
    assert "HTTP_PROXY" not in service["environment"]


def _write_fake_command(path: Path, body: str) -> None:
    path.write_text("#!/bin/sh\nset -eu\n" + body)
    path.chmod(0o755)


def _run_host_function(
    repo: Path,
    tmp_path: Path,
    invocation: str,
    *,
    extra_env: dict[str, str] | None = None,
    ps_output: str = "",
    ps_status: int = 0,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "docker.log"
    docker_body = r'''
{
  printf 'PWD=%s' "$PWD"
  for argument in "$@"; do printf '\t%s' "$argument"; done
  printf '\n'
} >> "$FAKE_DOCKER_LOG"

if [ -n "${OPENPI_BUILD_HTTP_PROXY:-}" ]; then
  printf '%s\n%s\n%s\n%s\n' \
    "$OPENPI_BUILD_HTTP_PROXY" \
    "$OPENPI_BUILD_HTTPS_PROXY" \
    "${OPENPI_BUILD_ALL_PROXY:-}" \
    "${OPENPI_BUILD_NO_PROXY:-}" > "$FAKE_PROXY_LOG"
fi

for argument in "$@"; do
  if [ "$argument" = ps ]; then
    if [ "${FAKE_PS_STATUS:-0}" -ne 0 ]; then
      exit "$FAKE_PS_STATUS"
    fi
    printf '%b' "${FAKE_PS_OUTPUT:-}"
    break
  fi
done
'''
    _write_fake_command(bin_dir / "docker", docker_body)

    data_dir = tmp_path / "openpi-data"
    env = {
        **os.environ,
        "FAKE_DOCKER_LOG": str(log_path),
        "FAKE_PROXY_LOG": str(tmp_path / "proxy.log"),
        "FAKE_PS_OUTPUT": ps_output,
        "FAKE_PS_STATUS": str(ps_status),
        "OPENPI_DOCKER_DATA_DIR": str(data_dir),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
    }
    for key in (
        "ALL_PROXY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "all_proxy",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    ):
        env.pop(key, None)
    if extra_env is not None:
        env.update(extra_env)
    source = shlex.quote(str(repo / "functions.zsh"))
    result = subprocess.run(
        ["zsh", "-fc", f"source {source} || exit; {invocation}"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    calls = []
    if log_path.exists():
        calls = [line.split("\t") for line in log_path.read_text().splitlines()]
    return result, calls, data_dir


def test_sourcing_host_functions_has_no_docker_side_effect(repo: Path, tmp_path: Path) -> None:
    result, calls, _ = _run_host_function(repo, tmp_path, ":")

    assert result.returncode == 0
    assert calls == []


def test_host_build_resolves_paths_from_the_sourced_file(repo: Path, tmp_path: Path) -> None:
    result, calls, data_dir = _run_host_function(repo, tmp_path, "opid-build")

    assert result.returncode == 0, result.stderr
    assert calls == [
        [
            f"PWD={repo}",
            "compose",
            "-f",
            str(repo / "compose.yaml"),
            "build",
            "openpi",
        ]
    ]
    for directory in ("assets", "cache", "checkpoints", "data", "wandb"):
        assert (data_dir / directory).is_dir()


def test_host_build_maps_proxyon_loopback_to_the_build_gateway(repo: Path, tmp_path: Path) -> None:
    result, calls, _ = _run_host_function(
        repo,
        tmp_path,
        "opid-build",
        extra_env={
            "http_proxy": "http://127.0.0.1:17891",
            "https_proxy": "http://127.0.0.1:17891",
            "all_proxy": "socks5://127.0.0.1:17891",
            "no_proxy": "localhost,127.0.0.1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert calls[0][1:] == [
        "compose",
        "-f",
        str(repo / "compose.yaml"),
        "-f",
        str(repo / "compose.proxy.yaml"),
        "build",
        "openpi",
    ]
    assert (tmp_path / "proxy.log").read_text().splitlines() == [
        "http://host.docker.internal:17891",
        "http://host.docker.internal:17891",
        "socks5://host.docker.internal:17891",
        "localhost,127.0.0.1",
    ]


def test_host_build_rejects_proxy_credentials(repo: Path, tmp_path: Path) -> None:
    result, calls, _ = _run_host_function(
        repo,
        tmp_path,
        "opid-build",
        extra_env={"HTTPS_PROXY": "http://user:secret@proxy.example:8080"},
    )

    assert result.returncode == 2
    assert "Proxy URLs with credentials are not supported" in result.stderr
    assert calls == []


def test_host_gpu_up_uses_the_gpu_override(repo: Path, tmp_path: Path) -> None:
    result, calls, _ = _run_host_function(repo, tmp_path, "opid-gpu-up")

    assert result.returncode == 0, result.stderr
    assert calls[0][1:] == [
        "compose",
        "-f",
        str(repo / "compose.yaml"),
        "-f",
        str(repo / "compose.gpu.yaml"),
        "up",
        "-d",
        "openpi",
    ]


def test_host_shell_executes_only_when_one_container_is_running(repo: Path, tmp_path: Path) -> None:
    result, calls, _ = _run_host_function(repo, tmp_path, "opid-shell", ps_output="container-123\n")

    assert result.returncode == 0, result.stderr
    assert calls[0][1:][-5:] == ["ps", "--status", "running", "-q", "openpi"]
    assert calls[1][1:] == [
        "exec",
        "-it",
        "-w",
        "/workspace",
        "container-123",
        "/usr/bin/zsh",
        "-l",
    ]


@pytest.mark.parametrize(
    ("ps_output", "ps_status", "expected_status", "message"),
    [
        ("", 0, 1, "No running openpi container"),
        ("one\ntwo\n", 0, 1, "Multiple running openpi containers"),
        ("", 7, 7, "Could not query running openpi containers"),
    ],
)
def test_host_shell_refuses_ambiguous_or_failed_lookup(
    repo: Path,
    tmp_path: Path,
    ps_output: str,
    ps_status: int,
    expected_status: int,
    message: str,
) -> None:
    result, calls, _ = _run_host_function(repo, tmp_path, "opid-shell", ps_output=ps_output, ps_status=ps_status)

    assert result.returncode == expected_status
    assert message in result.stderr
    assert len(calls) == 1


def test_container_test_shortcut_runs_locked_environment(repo: Path, tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "uv.log"
    _write_fake_command(bin_dir / "uv", 'printf \'%s\\n\' "$*" >> "$FAKE_UV_LOG"\n')
    env = {
        **os.environ,
        "FAKE_UV_LOG": str(log_path),
        "OPENPI_CONTAINER_WORKSPACE": str(repo),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
    }
    functions = shlex.quote(str(repo / "scripts/docker/container-functions.zsh"))

    result = subprocess.run(
        ["zsh", "-fc", f"source {functions}; opic-test -q"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert log_path.read_text().strip() == "run --no-sync pytest --strict-markers -m not manual -q"


def test_dependency_sync_retries_transient_uv_failures(repo: Path, tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    count_path = tmp_path / "uv-count"
    args_path = tmp_path / "uv-args"
    uv_body = r'''
count=0
if [ -r "$FAKE_UV_COUNT" ]; then count="$(cat "$FAKE_UV_COUNT")"; fi
count=$((count + 1))
printf '%s\n' "$count" > "$FAKE_UV_COUNT"
printf '%s\n' "$*" >> "$FAKE_UV_ARGS"
[ "$count" -ge 3 ]
'''
    _write_fake_command(bin_dir / "uv", uv_body)
    env = {
        **os.environ,
        "FAKE_UV_ARGS": str(args_path),
        "FAKE_UV_COUNT": str(count_path),
        "OPENPI_UV_RETRY_DELAY_SECONDS": "0",
        "OPENPI_UV_SYNC_ATTEMPTS": "3",
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
    }

    result = subprocess.run(
        ["sh", str(repo / "scripts/docker/sync-dependencies.sh")],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert count_path.read_text().strip() == "3"
    assert args_path.read_text().splitlines() == [
        "sync --frozen --no-install-project --dev",
        "sync --frozen --no-install-project --dev",
        "sync --frozen --no-install-project --dev",
    ]
    assert "uv sync failed (attempt 2/3); retrying" in result.stderr


def test_python_venv_creation_retries_transient_download_failures(repo: Path, tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    count_path = tmp_path / "uv-count"
    args_path = tmp_path / "uv-args"
    uv_body = r'''
count=0
if [ -r "$FAKE_UV_COUNT" ]; then count="$(cat "$FAKE_UV_COUNT")"; fi
count=$((count + 1))
printf '%s\n' "$count" > "$FAKE_UV_COUNT"
printf '%s|timeout=%s|mirror=%s\n' \
  "$*" "${UV_HTTP_TIMEOUT:-}" "${UV_PYTHON_INSTALL_MIRROR:-}" >> "$FAKE_UV_ARGS"
[ "$count" -ge 3 ]
'''
    _write_fake_command(bin_dir / "uv", uv_body)
    env = {
        **os.environ,
        "FAKE_UV_ARGS": str(args_path),
        "FAKE_UV_COUNT": str(count_path),
        "OPENPI_UV_PYTHON_ATTEMPTS": "3",
        "OPENPI_UV_RETRY_DELAY_SECONDS": "0",
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "UV_HTTP_TIMEOUT": "420",
        "UV_PROJECT_ENVIRONMENT": "/test-venv",
        "UV_PYTHON_INSTALL_MIRROR": "https://python-download.example/releases/download",
    }

    result = subprocess.run(
        ["sh", str(repo / "scripts/docker/create-venv.sh")],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert count_path.read_text().strip() == "3"
    assert args_path.read_text().splitlines() == [
        (
            "venv --python 3.11.9 /test-venv|timeout=420|"
            "mirror=https://python-download.example/releases/download"
        )
    ] * 3
    assert "uv venv failed (attempt 2/3); retrying" in result.stderr
