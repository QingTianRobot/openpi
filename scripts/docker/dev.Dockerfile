# syntax=docker/dockerfile:1.7

# Development image for the repository-level Compose workflow.
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0

COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

ARG DEBIAN_FRONTEND=noninteractive
ARG UBUNTU_APT_MIRROR=https://archive.ubuntu.com/ubuntu
ARG UBUNTU_SECURITY_MIRROR=https://security.ubuntu.com/ubuntu
RUN set -eu; \
    for mirror in "${UBUNTU_APT_MIRROR}" "${UBUNTU_SECURITY_MIRROR}"; do \
        printf '%s\n' "${mirror}" | LC_ALL=C grep -Eq '^https://[^@&|\\]+$' || \
            { echo "APT mirror must be a credential-free HTTPS URL" >&2; exit 1; }; \
    done; \
    sed -i \
        -e "s|https\\?://archive\\.ubuntu\\.com/ubuntu|${UBUNTU_APT_MIRROR}|g" \
        -e "s|https\\?://security\\.ubuntu\\.com/ubuntu|${UBUNTU_SECURITY_MIRROR}|g" \
        /etc/apt/sources.list; \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        ffmpeg \
        git \
        git-lfs \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        linux-headers-generic \
        zsh && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

ENV UV_LINK_MODE=copy
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
ENV UV_PROJECT_ENVIRONMENT=/.venv
ENV PATH="/.venv/bin:${PATH}"
ENV PYTHONPATH=/workspace/src:/workspace/packages/openpi-client/src

ARG UV_HTTP_TIMEOUT=300
ARG UV_PYTHON_INSTALL_MIRROR=https://github.com/indygreg/python-build-standalone/releases/download
ARG OPENPI_UV_PYTHON_ATTEMPTS=3
ARG OPENPI_UV_RETRY_DELAY_SECONDS=5
COPY --chmod=0755 scripts/docker/create-venv.sh /usr/local/bin/openpi-create-venv
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT}" \
    UV_PYTHON_INSTALL_MIRROR="${UV_PYTHON_INSTALL_MIRROR}" \
    OPENPI_UV_PYTHON_ATTEMPTS="${OPENPI_UV_PYTHON_ATTEMPTS}" \
    OPENPI_UV_RETRY_DELAY_SECONDS="${OPENPI_UV_RETRY_DELAY_SECONDS}" \
    openpi-create-venv

# Install locked third-party and development dependencies before copying source
# so application changes do not invalidate this expensive layer.
ARG UV_INDEX_URL=https://pypi.org/simple
ARG OPENPI_UV_SYNC_ATTEMPTS=3
COPY --chmod=0755 scripts/docker/sync-dependencies.sh /usr/local/bin/openpi-sync-dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/pyproject.toml,target=packages/openpi-client/pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/src,target=packages/openpi-client/src \
    UV_INDEX_URL="${UV_INDEX_URL}" \
    UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT}" \
    OPENPI_UV_SYNC_ATTEMPTS="${OPENPI_UV_SYNC_ATTEMPTS}" \
    OPENPI_UV_RETRY_DELAY_SECONDS="${OPENPI_UV_RETRY_DELAY_SECONDS}" \
    openpi-sync-dependencies

# Apply the repository's required PyTorch transformers compatibility patch.
COPY src/openpi/models_pytorch/transformers_replace/ /tmp/transformers_replace/
RUN transformers_dir="$(/.venv/bin/python -c 'import pathlib, transformers; print(pathlib.Path(transformers.__file__).parent)')" && \
    cp -r /tmp/transformers_replace/. "${transformers_dir}/" && \
    rm -rf /tmp/transformers_replace

ARG RUNTIME_USER=openpi
ARG RUNTIME_UID=1000
ARG RUNTIME_GID=1000
RUN set -eu; \
    group_owner="$(getent group "${RUNTIME_GID}" | cut -d: -f1 || true)"; \
    user_owner="$(getent passwd "${RUNTIME_UID}" | cut -d: -f1 || true)"; \
    [ -z "${group_owner}" ] || { echo "requested GID ${RUNTIME_GID} belongs to ${group_owner}" >&2; exit 1; }; \
    [ -z "${user_owner}" ] || { echo "requested UID ${RUNTIME_UID} belongs to ${user_owner}" >&2; exit 1; }; \
    groupadd --gid "${RUNTIME_GID}" "${RUNTIME_USER}"; \
    useradd --create-home --uid "${RUNTIME_UID}" --gid "${RUNTIME_GID}" --shell /usr/bin/zsh "${RUNTIME_USER}"; \
    mkdir -p /etc/openpi/zsh /usr/local/share/openpi /workspace; \
    chown "${RUNTIME_UID}:${RUNTIME_GID}" /workspace

COPY --chmod=0644 scripts/docker/container-functions.zsh /usr/local/share/openpi/functions.zsh
COPY --chmod=0644 scripts/docker/container.zshenv /etc/openpi/zsh/.zshenv
COPY --chmod=0644 scripts/docker/container.zshrc /etc/openpi/zsh/.zshrc

ENV HOME=/home/openpi
ENV SHELL=/usr/bin/zsh
ENV ZDOTDIR=/etc/openpi/zsh

USER ${RUNTIME_USER}

CMD ["sleep", "infinity"]
