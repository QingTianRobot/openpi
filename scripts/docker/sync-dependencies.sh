#!/bin/sh

set -u

attempts="${OPENPI_UV_SYNC_ATTEMPTS:-3}"
delay_seconds="${OPENPI_UV_RETRY_DELAY_SECONDS:-5}"

case "${attempts}" in
  ''|*[!0-9]*|0)
    echo "OPENPI_UV_SYNC_ATTEMPTS must be a positive integer" >&2
    exit 2
    ;;
esac
case "${delay_seconds}" in
  ''|*[!0-9]*)
    echo "OPENPI_UV_RETRY_DELAY_SECONDS must be a non-negative integer" >&2
    exit 2
    ;;
esac

attempt=1
while :; do
  if GIT_LFS_SKIP_SMUDGE=1 \
      GIT_CONFIG_COUNT=1 \
      GIT_CONFIG_KEY_0=http.version \
      GIT_CONFIG_VALUE_0=HTTP/1.1 \
      uv sync --frozen --no-install-project --dev; then
    exit 0
  else
    sync_status=$?
  fi

  if [ "${attempt}" -ge "${attempts}" ]; then
    echo "uv sync failed after ${attempts} attempts" >&2
    exit "${sync_status}"
  fi

  echo "uv sync failed (attempt ${attempt}/${attempts}); retrying" >&2
  sleep "${delay_seconds}"
  attempt=$((attempt + 1))
done
