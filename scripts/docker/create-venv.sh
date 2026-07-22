#!/bin/sh

set -u

attempts="${OPENPI_UV_PYTHON_ATTEMPTS:-3}"
delay_seconds="${OPENPI_UV_RETRY_DELAY_SECONDS:-5}"
python_version="${OPENPI_PYTHON_VERSION:-3.11.9}"
venv_path="${UV_PROJECT_ENVIRONMENT:-/.venv}"
python_mirror="${UV_PYTHON_INSTALL_MIRROR:-}"

case "${attempts}" in
  ''|*[!0-9]*|0)
    echo "OPENPI_UV_PYTHON_ATTEMPTS must be a positive integer" >&2
    exit 2
    ;;
esac
case "${delay_seconds}" in
  ''|*[!0-9]*)
    echo "OPENPI_UV_RETRY_DELAY_SECONDS must be a non-negative integer" >&2
    exit 2
    ;;
esac
case "${python_mirror}" in
  https://*) ;;
  *)
    echo "UV_PYTHON_INSTALL_MIRROR must be a credential-free HTTPS URL" >&2
    exit 2
    ;;
esac
case "${python_mirror}" in
  *'@'*|*'&'*|*'|'*|*'\'*)
    echo "UV_PYTHON_INSTALL_MIRROR must be a credential-free HTTPS URL" >&2
    exit 2
    ;;
esac

attempt=1
while :; do
  if uv venv --python "${python_version}" "${venv_path}"; then
    exit 0
  else
    venv_status=$?
  fi

  if [ "${attempt}" -ge "${attempts}" ]; then
    echo "uv venv failed after ${attempts} attempts" >&2
    exit "${venv_status}"
  fi

  echo "uv venv failed (attempt ${attempt}/${attempts}); retrying" >&2
  sleep "${delay_seconds}"
  attempt=$((attempt + 1))
done
