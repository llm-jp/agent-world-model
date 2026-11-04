#!/usr/bin/env bash

set -eu -o pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The job body treats the submit directory as the repository root,
# so submit from the root wherever this script is called from.
cd "${REPO_DIR}"

USAGE_TEXT=$(
	cat <<'USAGE'
Usage: installers/setup.sh [options]

Options:
  --platform <abci|mdx>   cluster to set up (required)
  --local                 run the setup here instead of submitting a job
  -h, --help              show this help

Example:
  installers/setup.sh --platform abci
USAGE
)

PLATFORM=""
LOCAL_RUN=0

while [ $# -gt 0 ]; do
	case "$1" in
	--platform)
		PLATFORM=$2
		shift 2
		;;
	--local)
		LOCAL_RUN=1
		shift
		;;
	-h | --help)
		echo "${USAGE_TEXT}"
		exit 0
		;;
	*)
		>&2 echo "Unknown argument: $1"
		>&2 echo "${USAGE_TEXT}"
		exit 1
		;;
	esac
done

: "${PLATFORM:?--platform is required (abci or mdx).}"

export PLATFORM

if [ "${LOCAL_RUN}" = "1" ]; then
	exec bash installers/job_setup.sh
fi

mkdir -p logs

case "${PLATFORM}" in
abci)
	# QSUB_EXTRA_ARGS is split into words on purpose to pass several arguments.
	# shellcheck disable=SC2086
	exec qsub \
		-P gcg51557 \
		-q R9920261000 \
		-N 0375_awm_install \
		-l select=1:ncpus=8:mem=160gb \
		-l walltime=03:00:00 \
		-v "RTYPE=${RTYPE:-rt_HF},PLATFORM=abci" \
		-o /dev/null -e /dev/null \
		-m n \
		${QSUB_EXTRA_ARGS:-} \
		installers/job_setup.sh
	;;
mdx)
	# SBATCH_EXTRA_ARGS is split into words on purpose to pass several arguments.
	# shellcheck disable=SC2086
	exec sbatch \
		--job-name=0375_awm_install \
		--partition=gpu \
		--nodes=1 \
		--ntasks=1 \
		--cpus-per-task=8 \
		--gres=gpu:1 \
		--mem=48G \
		--time=03:00:00 \
		--output=logs/install-%j.out \
		--error=logs/install-%j.out \
		--export=ALL \
		${SBATCH_EXTRA_ARGS:-} \
		installers/job_setup.sh
	;;
*)
	>&2 echo "Unknown platform: ${PLATFORM} (expected abci or mdx)"
	exit 1
	;;
esac
