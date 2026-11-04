#!/usr/bin/env bash

set -eu -o pipefail

REPO_DIR=${PBS_O_WORKDIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
cd "${REPO_DIR}"

: "${PLATFORM:?PLATFORM is required (set by installers/setup.sh).}"

# ABCI jobs save their own log, since qsub is given -o /dev/null.
if [ -n "${PBS_JOBID:-}" ]; then
	mkdir -p logs
	exec >"logs/install-${PBS_JOBID%%.*}.out" 2>&1
fi

# uv copies packages from its cache, which may sit on another file system.
export UV_LINK_MODE=copy

EXP_ROOT="$(pwd -P)"
export TMPDIR="${EXP_ROOT}/.tmp"
mkdir -p "${TMPDIR}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] platform=${PLATFORM} root=${EXP_ROOT}"

# uv sync downloads the interpreter here instead of under the home directory,
# and recreates a venv that still points to another interpreter.
# The cluster's own Python is not used on either platform (see [tool.uv] in pyproject.toml).
export UV_PYTHON_INSTALL_DIR="${EXP_ROOT}/.python"

# pipeline environment (awm CLI)
uv sync

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${0}"
