#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL="https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"

if [[ -L "$ROOT/data" ]]; then
  echo "Refusing to write downloaded data through the existing data symlink." >&2
  exit 1
fi

if [[ -d "$ROOT/data" ]] && [[ -n "$(find "$ROOT/data" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite a non-empty data directory." >&2
  exit 1
fi

TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT

curl --fail --location --retry 3 --silent --show-error "$URL" --output "$TEMP_DIR/cmapss.zip"
mkdir -p "$TEMP_DIR/extracted"
unzip -q "$TEMP_DIR/cmapss.zip" -d "$TEMP_DIR/extracted"

DATASET_DIR="$(find "$TEMP_DIR/extracted" -type d -name CMAPSSData -print -quit)"
if [[ -z "$DATASET_DIR" ]]; then
  echo "Could not find the nested CMAPSSData directory in the archive." >&2
  exit 1
fi

for file in train_FD001.txt test_FD001.txt RUL_FD001.txt train_FD004.txt test_FD004.txt RUL_FD004.txt; do
  if [[ ! -f "$DATASET_DIR/$file" ]]; then
    echo "Missing expected dataset file: $file" >&2
    exit 1
  fi
done

mkdir -p "$ROOT/data"
cp "$DATASET_DIR"/*.txt "$ROOT/data/"
