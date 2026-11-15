#!/usr/bin/env bash
# Downloads MNIST into data/mnist and decompresses it (IDX format).
set -euo pipefail
DEST="${1:-data/mnist}"
mkdir -p "$DEST"
FILES="train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte"
MIRRORS="https://ossci-datasets.s3.amazonaws.com/mnist https://storage.googleapis.com/cvdf-datasets/mnist"

for f in $FILES; do
  if [ -f "$DEST/$f" ]; then echo "ok   $f"; continue; fi
  for m in $MIRRORS; do
    if curl -fsSL "$m/$f.gz" -o "$DEST/$f.gz"; then
      gunzip -f "$DEST/$f.gz"; echo "got  $f ($m)"; break
    fi
  done
  [ -f "$DEST/$f" ] || { echo "could not download $f"; exit 1; }
done
