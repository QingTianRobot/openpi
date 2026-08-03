# Source this file from any directory:
#   source /path/to/openpi/functions.zsh

typeset -g OPENPI_REPO_DIR="${${(%):-%N}:A:h}"
typeset -g OPENPI_COMPOSE_SERVICE="openpi"
typeset -g OPENPI_BASE_COMPOSE_FILE="${OPENPI_REPO_DIR}/compose.yaml"
typeset -g OPENPI_GPU_COMPOSE_FILE="${OPENPI_REPO_DIR}/compose.gpu.yaml"
typeset -g OPENPI_PROXY_COMPOSE_FILE="${OPENPI_REPO_DIR}/compose.proxy.yaml"
typeset -gx OPENPI_DOCKER_DATA_DIR="${OPENPI_DOCKER_DATA_DIR:-${OPENPI_REPO_DIR:h}/openpi-docker-data}"
typeset -gx OPENPI_COMPOSE_PROJECT_NAME="${OPENPI_COMPOSE_PROJECT_NAME:-openpi-dev-$(id -u)}"
typeset -ga OPENPI_DOCKER_HOST_DIRS=(
  "${OPENPI_DOCKER_DATA_DIR}/assets"
  "${OPENPI_DOCKER_DATA_DIR}/cache"
  "${OPENPI_DOCKER_DATA_DIR}/checkpoints"
  "${OPENPI_DOCKER_DATA_DIR}/data"
  "${OPENPI_DOCKER_DATA_DIR}/wandb"
)

_opid_prepare_host() {
  emulate -L zsh
  typeset -gx OPENPI_UID="$(id -u)"
  typeset -gx OPENPI_GID="$(id -g)"
  mkdir -p -- "${OPENPI_DOCKER_HOST_DIRS[@]}"
}

opid-download() {
  emulate -L zsh
  if (( $# < 1 || $# > 2 )); then
    print -u2 "Usage: opid-download <checkpoint-name-or-uri> [--force]"
    return 2
  fi

  local checkpoint="$1"
  local force_download=false
  if (( $# == 2 )); then
    if [[ "$2" != "--force" ]]; then
      print -u2 "Usage: opid-download <checkpoint-name-or-uri> [--force]"
      return 2
    fi
    force_download=true
  fi

  local checkpoint_uri="${checkpoint}"
  if [[ "${checkpoint}" != *://* ]]; then
    checkpoint_uri="gs://openpi-assets/checkpoints/${checkpoint}"
  fi

  local python_code="from openpi.shared import download; import sys; print(download.maybe_download(sys.argv[1], force_download=sys.argv[2] == 'true'))"
  (
    cd "${OPENPI_REPO_DIR}" || return 1
    _opid_prepare_host || return 1
    docker compose -f "${OPENPI_BASE_COMPOSE_FILE}" run --rm -T "${OPENPI_COMPOSE_SERVICE}" \
      uv run --no-sync python -c "${python_code}" "${checkpoint_uri}" "${force_download}"
  )
}

_opid_has_build_proxy() {
  [[ -n "${OPENPI_BUILD_HTTP_PROXY:-${http_proxy:-${HTTP_PROXY:-}}}" ||
     -n "${OPENPI_BUILD_HTTPS_PROXY:-${https_proxy:-${HTTPS_PROXY:-}}}" ]]
}

_opid_prepare_proxy_url() {
  emulate -L zsh
  local value="$1"
  local authority="${value#*://}"
  authority="${authority%%/*}"

  case "${value}" in
    http://*|https://*|socks://*|socks5://*|socks5h://*) ;;
    *)
      print -u2 "Unsupported proxy URL: expected http(s):// or socks5://."
      return 2
      ;;
  esac
  if [[ "${authority}" == *"@"* ]]; then
    print -u2 "Proxy URLs with credentials are not supported in Docker build arguments."
    return 2
  fi

  value="${value//127.0.0.1/host.docker.internal}"
  value="${value//localhost/host.docker.internal}"
  reply="${value}"
}

_opid_prepare_build_proxy() {
  emulate -L zsh
  local raw_http="${OPENPI_BUILD_HTTP_PROXY:-${HTTP_PROXY:-${http_proxy:-}}}"
  local raw_https="${OPENPI_BUILD_HTTPS_PROXY:-${HTTPS_PROXY:-${https_proxy:-}}}"
  local raw_all="${OPENPI_BUILD_ALL_PROXY:-${ALL_PROXY:-${all_proxy:-}}}"

  [[ -n "${raw_http}" ]] || raw_http="${raw_https}"
  [[ -n "${raw_https}" ]] || raw_https="${raw_http}"

  _opid_prepare_proxy_url "${raw_http}" || return
  typeset -gx OPENPI_BUILD_HTTP_PROXY="${reply}"
  _opid_prepare_proxy_url "${raw_https}" || return
  typeset -gx OPENPI_BUILD_HTTPS_PROXY="${reply}"
  if [[ -n "${raw_all}" ]]; then
    _opid_prepare_proxy_url "${raw_all}" || return
    typeset -gx OPENPI_BUILD_ALL_PROXY="${reply}"
  else
    typeset -gx OPENPI_BUILD_ALL_PROXY=""
  fi
  typeset -gx OPENPI_BUILD_NO_PROXY="${OPENPI_BUILD_NO_PROXY:-${NO_PROXY:-${no_proxy:-}}}"
}

opid-build() {
  (
    cd "${OPENPI_REPO_DIR}" || return 1
    _opid_prepare_host || return 1
    if _opid_has_build_proxy; then
      _opid_prepare_build_proxy || return
      print "Using the current shell proxy for this Docker build."
      docker compose \
        -f "${OPENPI_BASE_COMPOSE_FILE}" \
        -f "${OPENPI_PROXY_COMPOSE_FILE}" \
        build "${OPENPI_COMPOSE_SERVICE}"
    else
      docker compose -f "${OPENPI_BASE_COMPOSE_FILE}" build "${OPENPI_COMPOSE_SERVICE}"
    fi
  )
}

opid-up() {
  (
    cd "${OPENPI_REPO_DIR}" || return 1
    _opid_prepare_host || return 1
    docker compose -f "${OPENPI_BASE_COMPOSE_FILE}" up -d "${OPENPI_COMPOSE_SERVICE}"
  )
}

opid-gpu-up() {
  (
    cd "${OPENPI_REPO_DIR}" || return 1
    _opid_prepare_host || return 1
    docker compose \
      -f "${OPENPI_BASE_COMPOSE_FILE}" \
      -f "${OPENPI_GPU_COMPOSE_FILE}" \
      up -d "${OPENPI_COMPOSE_SERVICE}"
  )
}

opid-run() {
  (
    cd "${OPENPI_REPO_DIR}" || return 1
    _opid_prepare_host || return 1
    docker compose -f "${OPENPI_BASE_COMPOSE_FILE}" run --rm "${OPENPI_COMPOSE_SERVICE}" /usr/bin/zsh -l
  )
}

opid-gpu-run() {
  (
    cd "${OPENPI_REPO_DIR}" || return 1
    _opid_prepare_host || return 1
    docker compose \
      -f "${OPENPI_BASE_COMPOSE_FILE}" \
      -f "${OPENPI_GPU_COMPOSE_FILE}" \
      run --rm "${OPENPI_COMPOSE_SERVICE}" /usr/bin/zsh -l
  )
}

_opid_running_ids() {
  emulate -L zsh
  local output compose_status
  output="$(
    cd "${OPENPI_REPO_DIR}" || return 1
    _opid_prepare_host || return 1
    docker compose -f "${OPENPI_BASE_COMPOSE_FILE}" ps --status running -q "${OPENPI_COMPOSE_SERVICE}"
  )"
  compose_status=$?
  if (( compose_status != 0 )); then
    print -u2 "Could not query running openpi containers (status ${compose_status})."
    return "${compose_status}"
  fi
  reply=("${(@f)output}")
  reply=("${(@)reply:#}")
}

opid-shell() {
  emulate -L zsh
  local -a ids
  _opid_running_ids || return
  ids=("${reply[@]}")
  case "${#ids}" in
    0)
      print -u2 "No running openpi container. Start one with opid-up or opid-gpu-up."
      return 1
      ;;
    1)
      (
        cd "${OPENPI_REPO_DIR}" || return 1
        docker exec -it -w /workspace "${ids[1]}" /usr/bin/zsh -l
      )
      ;;
    *)
      print -u2 "Multiple running openpi containers; refusing to choose:"
      local id
      for id in "${ids[@]}"; do
        print -u2 -- "  ${id}"
      done
      return 1
      ;;
  esac
}

opid-ps() {
  (
    cd "${OPENPI_REPO_DIR}" || return 1
    _opid_prepare_host || return 1
    docker compose -f "${OPENPI_BASE_COMPOSE_FILE}" ps
  )
}

opid-logs() {
  (
    cd "${OPENPI_REPO_DIR}" || return 1
    _opid_prepare_host || return 1
    docker compose -f "${OPENPI_BASE_COMPOSE_FILE}" logs -f --tail="${OPENPI_LOG_TAIL:-200}" "$@"
  )
}

opid-down() {
  (
    cd "${OPENPI_REPO_DIR}" || return 1
    _opid_prepare_host || return 1
    docker compose -f "${OPENPI_BASE_COMPOSE_FILE}" down
  )
}

opid-config() {
  (
    cd "${OPENPI_REPO_DIR}" || return 1
    _opid_prepare_host || return 1
    docker compose -f "${OPENPI_BASE_COMPOSE_FILE}" config
  )
}

opid-gpu-config() {
  (
    cd "${OPENPI_REPO_DIR}" || return 1
    _opid_prepare_host || return 1
    docker compose -f "${OPENPI_BASE_COMPOSE_FILE}" -f "${OPENPI_GPU_COMPOSE_FILE}" config
  )
}

opid-help() {
  print "OpenPI host Docker commands (opid-*):"
  print "  opid-build       Build the image; use the current shell proxy when set"
  print "  opid-up          Start the CPU development container in the background"
  print "  opid-gpu-up      Start with GPU OPENPI_GPU_DEVICE (default: 0)"
  print "  opid-run         Open a disposable CPU container shell"
  print "  opid-gpu-run     Open a disposable GPU container shell"
  print "  opid-download <name-or-uri> [--force] Download and cache model weights"
  print "  opid-shell       Enter the single running development container"
  print "  opid-ps          Show container status"
  print "  opid-logs        Follow container logs"
  print "  opid-down        Stop and remove the Compose container"
  print "  opid-config      Render the base Compose configuration"
  print "  opid-gpu-config  Render the GPU Compose configuration"
}
