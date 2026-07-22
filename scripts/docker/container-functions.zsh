typeset -g OPENPI_CONTAINER_WORKSPACE="${OPENPI_CONTAINER_WORKSPACE:-/workspace}"

_opic_in_workspace() {
  (
    cd "${OPENPI_CONTAINER_WORKSPACE}" || {
      print -u2 "OpenPI workspace not found: ${OPENPI_CONTAINER_WORKSPACE}"
      return 1
    }
    command "$@"
  )
}

opic-test() {
  _opic_in_workspace uv run --no-sync pytest --strict-markers -m "not manual" "$@"
}

opic-python() {
  _opic_in_workspace uv run --no-sync python "$@"
}

opic-norm() {
  _opic_in_workspace uv run --no-sync scripts/compute_norm_stats.py "$@"
}

opic-train() {
  if (( $# == 0 )); then
    print -u2 "Usage: opic-train <config-name> [training arguments...]"
    return 2
  fi
  (
    cd "${OPENPI_CONTAINER_WORKSPACE}" || return 1
    XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
      uv run --no-sync scripts/train.py "$@"
  )
}

opic-serve() {
  _opic_in_workspace uv run --no-sync scripts/serve_policy.py "$@"
}

opic-help() {
  print "OpenPI container commands (opic-*):"
  print "  opic-test [args]         Run the non-manual pytest suite"
  print "  opic-python [args]       Run Python in the locked environment"
  print "  opic-norm [args]         Compute dataset normalization statistics"
  print "  opic-train <config> ...  Start JAX training (90% GPU memory by default)"
  print "  opic-serve [args]        Start the policy server on port 8000"
}
