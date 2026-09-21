#!/usr/bin/env bash
# Build multi-platform release artifacts + SHA256SUMS for hearim.
# Follows the ziozzang/hftools release scheme consumed by `hearim update`.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${ROOT_DIR}/dist"
VERSION="${VERSION:?set VERSION=0.2.0}"

if [[ ! "${VERSION}" =~ ^[0-9A-Za-z._-]+$ ]]; then
  echo "invalid VERSION: ${VERSION}" >&2
  exit 1
fi

mkdir -p "${DIST_DIR}"

targets=(
  "darwin arm64 macos arm64"
  "darwin amd64 macos x86_64"
  "windows arm64 windows arm64"
  "windows amd64 windows x86_64"
  "linux arm64 linux arm64"
  "linux amd64 linux x86_64"
)

artifacts=()
for target in "${targets[@]}"; do
  read -r goos goarch platform arch <<<"${target}"
  extension=""
  if [[ "${goos}" == "windows" ]]; then
    extension=".exe"
  fi
  filename="hearim_${VERSION}_${platform}_${arch}${extension}"
  output="${DIST_DIR}/${filename}"
  echo "building ${filename}"
  (
    cd "${ROOT_DIR}"
    CGO_ENABLED=0 GOOS="${goos}" GOARCH="${goarch}" \
      go build -buildvcs=false -trimpath -tags="netgo,osusergo" -ldflags="-s -w" -o "${output}" ./cmd/hearim
  )
  if [[ "${goos}" == "windows" ]]; then
    chmod 0644 "${output}"
  else
    chmod 0755 "${output}"
  fi
  artifacts+=("${filename}")
done

(
  cd "${DIST_DIR}"
  sha256sum "${artifacts[@]}" > SHA256SUMS
  chmod 0644 SHA256SUMS
)
echo "release artifacts in ${DIST_DIR}"
