#!/usr/bin/env bash
# One-shot setup: Intrinsic_assets (if missing), Docker image (if missing), start container, enter shell.
# Optional: export ISAACLAB_ROOT="$HOME/IsaacLab"
#
# Isaac Lab layout:
#   .../aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/Intrinsic_assets/

set -euo pipefail

ZIP_URL="https://developer.nvidia.com/downloads/Omniverse/learning/Events/Hackathons/Intrinsic_assets.zip"

read_docker_name_suffix() {
  local f="${ISAACLAB_ROOT}/docker/.env.base"
  [[ -f "$f" ]] || { echo ""; return 0; }
  grep '^DOCKER_NAME_SUFFIX=' "$f" | head -1 | cut -d= -f2- | tr -d '\r' | sed 's/^"\(.*\)"$/\1/'
}

docker_container_status() {
  docker container inspect -f '{{.State.Status}}' "$1" 2>/dev/null | tr -d '\r\n' || true
}

if [[ -z "${ISAACLAB_ROOT:-}" ]]; then
  if [[ -d "$HOME/IsaacLab" ]]; then
    ISAACLAB_ROOT="$HOME/IsaacLab"
  else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ISAACLAB_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  fi
fi

DEST_PARENT="${ISAACLAB_ROOT}/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task"
DEST_ASSETS="${DEST_PARENT}/Intrinsic_assets"

if [[ ! -d "$DEST_PARENT" ]]; then
  echo "error: task directory missing — set ISAACLAB_ROOT to Isaac Lab root (parent of aic/):" >&2
  echo "  $DEST_PARENT" >&2
  exit 1
fi

if [[ -d "$DEST_ASSETS" ]] && [[ -n "$(ls -A "$DEST_ASSETS" 2>/dev/null || true)" ]]; then
  echo "Intrinsic_assets already present — skipping download."
else
  echo "Downloading Intrinsic_assets ..."
  rm -rf "$DEST_ASSETS"
  TMP_ZIP="$(mktemp "${TMPDIR:-/tmp}/intrinsic_assets.XXXXXX.zip")"
  trap 'rm -f "$TMP_ZIP"' EXIT
  curl -fsSL "$ZIP_URL" -o "$TMP_ZIP"
  unzip -q "$TMP_ZIP" -d "$DEST_PARENT"
  trap - EXIT
  rm -f "$TMP_ZIP"
  if [[ ! -d "$DEST_ASSETS" ]]; then
    echo "error: extract failed, missing $DEST_ASSETS" >&2
    exit 1
  fi
  echo "Installed: $DEST_ASSETS"
fi

if [[ ! -x "${ISAACLAB_ROOT}/docker/container.py" ]]; then
  echo "error: Isaac Lab docker helper missing or not executable:" >&2
  echo "  ${ISAACLAB_ROOT}/docker/container.py" >&2
  exit 1
fi

suffix="$(read_docker_name_suffix)"
img="isaac-lab-base${suffix}:latest"
ctr="isaac-lab-base${suffix}"

if ! docker image inspect "$img" >/dev/null 2>&1; then
  echo "Docker image missing — building base ($img) ..."
  (cd "$ISAACLAB_ROOT" && ./docker/container.py build base)
else
  echo "Docker image present — skipping build ($img)."
fi

if [[ "$(docker_container_status "$ctr")" != "running" ]]; then
  echo "Starting container $ctr ..."
  (cd "$ISAACLAB_ROOT" && ./docker/container.py start base)
else
  echo "Container already running — skipping start ($ctr)."
fi

echo "Entering container (interactive) ..."
cd "$ISAACLAB_ROOT"
exec ./docker/container.py enter base
