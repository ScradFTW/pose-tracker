#!/usr/bin/env bash
# Pulls the third-party onnxruntime-web runtime (WASM binary + loader) into
# vendor/. Not committed to git -- it's a large prebuilt binary, easy to
# refetch, and unrelated to the model this repo actually trains.
set -euo pipefail
cd "$(dirname "$0")"

VERSION=1.19.2
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

npm pack "onnxruntime-web@${VERSION}" --silent --pack-destination "$TMP"
tar xzf "$TMP"/onnxruntime-web-*.tgz -C "$TMP"

mkdir -p vendor
cp "$TMP/package/dist/ort.min.js" vendor/
cp "$TMP/package/dist/ort-wasm-simd-threaded.wasm" vendor/
cp "$TMP/package/dist/ort-wasm-simd-threaded.mjs" vendor/

echo "onnxruntime-web ${VERSION} runtime -> vendor/"
echo "Now export the trained model: (cd ../.. && train/export_onnx.py) writes vendor/pose_net.onnx"
