# pose-tracker

Real-time webcam skeletal tracking. Unlike a typical "wire up MediaPipe"
demo, the pose model here is trained from scratch: a ResNet18-shaped
convolutional heatmap-regression net (~15.4M parameters, random init --
no ImageNet pretraining), trained on COCO 2017 person keypoints
(~125k person instances, 17 joints). Exported to ONNX and run entirely
client-side via `onnxruntime-web` (WASM, multi-threaded via
SharedArrayBuffer) -- no server, no GPU required, no video ever leaves
the browser.

Live at `https://bradjobe.dev/pose-tracker/`.

First attempt trained on Leeds Sports Pose (2,000 images, 14 joints) and
topped out around PCK@0.2 = 0.48 -- clearly a data-starved model. Moving
to COCO (~125k instances vs 2k) and a deeper ResNet18-style backbone
(15.4M params vs 700k) got val PCK@0.2 to 0.93 in just 8 epochs (~70 min
on an RTX 3060). The old `train/dataset.py` / `train/model.py` /
`train/train.py` (LSP) are kept for reference; `train/dataset_coco.py` /
`train/model_resnet.py` / `train/train_coco.py` are what's deployed now.

## Layout

```
train/
  prepare_coco.py       filters COCO person_keypoints annotations, downloads only
                         the images actually needed (not the full 18GB zip)
  dataset_coco.py        bbox-crop dataset loader + augmentation + heatmap targets
  model_resnet.py         ResNet18-shaped backbone (from scratch) + deconv head
  train_coco.py           training loop, PCK@0.2 metric, writes eval_report_coco.txt
  export_onnx_coco.py     exports the trained checkpoint to ONNX, verifies parity
  check_quality.py        renders a grid of val predictions vs ground truth
  dataset.py, model.py, train.py, export_onnx.py   earlier LSP-based attempt (kept for reference)
frontend/pose-tracker/
  index.html, style.css, app.js   the demo page (COCO 17-joint model; neck/chest
                                   are derived client-side, COCO has no such keypoints)
  fetch-vendor.sh                 pulls the onnxruntime-web runtime (not committed)
  vendor/pose_net.onnx            the trained model (committed -- it's the point)
```

## Reproducing

```
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision onnx onnxruntime scipy numpy pillow huggingface_hub tqdm requests

# annotations + only the images that actually contain labeled people
# (up to 45k train images; caps total download around a few GB, not 18GB)
mkdir -p data/coco && cd data/coco
curl -sL -o annotations.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -o annotations.zip "annotations/person_keypoints_*" && rm annotations.zip
cd ../..
python train/prepare_coco.py

python train/train_coco.py            # ~8+ epochs, ~9 min/epoch on a single GPU
python train/export_onnx_coco.py      # writes frontend/pose-tracker/vendor/pose_net.onnx
python train/check_quality.py         # sanity-check grid, saved to /tmp

cd frontend/pose-tracker && ./fetch-vendor.sh   # pulls onnxruntime-web
```

Then serve `frontend/pose-tracker/` as static files. The page needs
`Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` response headers (see the
nginx config on the server) for multi-threaded WASM -- without them it
still works, just single-threaded and ~5x slower.

## Honest limitations

Single person, wrist position rather than finger articulation, neck/chest
are derived (shoulder midpoint / neck-hip midpoint) rather than predicted.
Still rougher than a commercial pose library trained on datasets and
compute far beyond this project's scope.
