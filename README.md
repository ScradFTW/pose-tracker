# pose-tracker

Real-time webcam skeletal tracking. Unlike a typical "wire up MediaPipe"
demo, the pose model here is trained from scratch: a small (~700k
parameter) convolutional heatmap-regression net, trained on the
[Leeds Sports Pose](https://huggingface.co/datasets/LiuRunky/Leeds_Sports_Pose)
dataset (2,000 labeled images, 14 joints, no pretrained backbone). Exported
to ONNX and run entirely client-side via `onnxruntime-web` (WASM) -- no
server, no GPU required, no video ever leaves the browser.

Live at `https://bradjobe.dev/pose-tracker/`.

## Layout

```
train/
  dataset.py       LSP dataset loader + augmentation + heatmap targets
  model.py         the CNN (stem -> 2 stages -> upsample -> per-joint heatmap head)
  train.py         training loop, PCK@0.2 metric, writes eval_report.txt
  export_onnx.py   exports the trained checkpoint to ONNX, verifies parity
frontend/pose-tracker/
  index.html, style.css, app.js   the demo page
  fetch-vendor.sh                 pulls the onnxruntime-web runtime (not committed)
  vendor/pose_net.onnx            the trained model (committed -- it's the point)
```

## Reproducing

```
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision onnx onnxruntime scipy numpy pillow huggingface_hub tqdm

python3 -c "from huggingface_hub import snapshot_download; \
  snapshot_download(repo_id='LiuRunky/Leeds_Sports_Pose', repo_type='dataset', local_dir='data/lsp_raw')"

python train/train.py           # ~150 epochs, a few minutes on a single GPU
python train/export_onnx.py     # writes frontend/pose-tracker/vendor/pose_net.onnx

cd frontend/pose-tracker && ./fetch-vendor.sh   # pulls onnxruntime-web
```

Then serve `frontend/pose-tracker/` as static files.

## Honest limitations

Single person, no hands/face, 14 joints. Trained on 2,000 images total --
noticeably rougher than a production pose library trained on datasets
orders of magnitude larger. The point was training something real from
scratch, not matching commercial accuracy.
