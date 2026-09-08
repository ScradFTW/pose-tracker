const IMG_SIZE = 256;
const HEATMAP_SIZE = 64;
const HEATMAP_STRIDE = IMG_SIZE / HEATMAP_SIZE;
const NUM_JOINTS = 14;

// Must match train/dataset.py SKELETON exactly.
const SKELETON = [
  [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
  [2, 8], [3, 9],
  [8, 9],
  [8, 7], [7, 6], [9, 10], [10, 11],
  [8, 12], [9, 12], [12, 13],
];

const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const startBtn = document.getElementById("start-btn");
const stageMessage = document.getElementById("stage-message");
const engineBadge = document.getElementById("engine-badge");
const mirrorToggle = document.getElementById("toggle-mirror");
const skeletonToggle = document.getElementById("toggle-skeleton");
const pointsToggle = document.getElementById("toggle-points");

const statInfer = document.getElementById("stat-infer");
const statE2e = document.getElementById("stat-e2e");
const statFps = document.getElementById("stat-fps");
const statLandmarks = document.getElementById("stat-landmarks");
const statParams = document.getElementById("stat-params");
const statPck = document.getElementById("stat-pck");

statParams.textContent = "705,774";
statPck.textContent = "0.48";

let session = null;
let running = false;
let rafId = null;

const fpsWindow = [];

// Offscreen canvas used to resize the video frame to the model's 256x256 input.
const inputCanvas = document.createElement("canvas");
inputCanvas.width = IMG_SIZE;
inputCanvas.height = IMG_SIZE;
const inputCtx = inputCanvas.getContext("2d", { willReadFrequently: true });

applyMirror();
mirrorToggle.addEventListener("change", applyMirror);

function applyMirror() {
  const mirrored = mirrorToggle.checked;
  video.classList.toggle("mirrored", mirrored);
  canvas.classList.toggle("mirrored", mirrored);
}

async function loadModel() {
  engineBadge.textContent = "loading model…";
  engineBadge.className = "badge";

  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;

  session = await ort.InferenceSession.create("./vendor/pose_net.onnx", {
    executionProviders: ["wasm"],
  });

  engineBadge.textContent = "model ready (from-scratch CNN)";
  engineBadge.className = "badge ready";
}

startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  startBtn.textContent = "Starting…";
  try {
    await ensureCamera();
    if (!session) {
      await loadModel();
    }
    stageMessage.classList.add("hidden");
    start();
  } catch (err) {
    reportError(err);
    startBtn.disabled = false;
    startBtn.textContent = "Enable camera";
  }
});

async function ensureCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise((resolve) => {
    video.onloadedmetadata = () => resolve();
  });
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

function reportError(err) {
  console.error(err);
  engineBadge.textContent = "error — see console";
  engineBadge.className = "badge error";
  const p = document.createElement("p");
  p.className = "hint";
  p.textContent = err && err.message ? err.message : String(err);
  stageMessage.appendChild(p);
}

function start() {
  running = true;
  rafId = requestAnimationFrame(loop);
}

function preprocess() {
  inputCtx.drawImage(video, 0, 0, IMG_SIZE, IMG_SIZE);
  const { data } = inputCtx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);

  const chw = new Float32Array(3 * IMG_SIZE * IMG_SIZE);
  const plane = IMG_SIZE * IMG_SIZE;
  for (let i = 0; i < plane; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    chw[i] = (r - 0.5) / 0.5;
    chw[plane + i] = (g - 0.5) / 0.5;
    chw[plane * 2 + i] = (b - 0.5) / 0.5;
  }
  return new ort.Tensor("float32", chw, [1, 3, IMG_SIZE, IMG_SIZE]);
}

function decodeHeatmaps(heatmapTensor) {
  const data = heatmapTensor.data;
  const plane = HEATMAP_SIZE * HEATMAP_SIZE;
  const keypoints = [];
  for (let j = 0; j < NUM_JOINTS; j++) {
    let best = -Infinity;
    let bx = 0;
    let by = 0;
    const offset = j * plane;
    for (let y = 0; y < HEATMAP_SIZE; y++) {
      for (let x = 0; x < HEATMAP_SIZE; x++) {
        const v = data[offset + y * HEATMAP_SIZE + x];
        if (v > best) {
          best = v;
          bx = x;
          by = y;
        }
      }
    }
    keypoints.push({
      x: (bx * HEATMAP_STRIDE) / IMG_SIZE,
      y: (by * HEATMAP_STRIDE) / IMG_SIZE,
      score: best,
    });
  }
  return keypoints;
}

async function loop() {
  if (!running) return;

  const frameStart = performance.now();
  const inputTensor = preprocess();

  const inferStart = performance.now();
  const results = await session.run({ input: inputTensor });
  const inferMs = performance.now() - inferStart;

  const keypoints = decodeHeatmaps(results.heatmaps);

  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawSkeleton(keypoints);
  ctx.restore();

  const e2eMs = performance.now() - frameStart;

  const now = performance.now();
  fpsWindow.push(now);
  while (fpsWindow.length && fpsWindow[0] < now - 1000) fpsWindow.shift();

  statInfer.textContent = `${inferMs.toFixed(1)} ms`;
  statE2e.textContent = `${e2eMs.toFixed(1)} ms`;
  statFps.textContent = fpsWindow.length.toString();
  statLandmarks.textContent = `${NUM_JOINTS}`;

  rafId = requestAnimationFrame(loop);
}

function drawSkeleton(keypoints) {
  const w = canvas.width;
  const h = canvas.height;
  const pts = keypoints.map((kp) => ({ x: kp.x * w, y: kp.y * h, score: kp.score }));

  const CONFIDENT = 0.05; // heatmap peak threshold below which a joint is treated as not-found

  if (skeletonToggle.checked) {
    ctx.strokeStyle = "#4fd1c5";
    ctx.lineWidth = 3;
    for (const [a, b] of SKELETON) {
      if (pts[a].score < CONFIDENT || pts[b].score < CONFIDENT) continue;
      ctx.beginPath();
      ctx.moveTo(pts[a].x, pts[a].y);
      ctx.lineTo(pts[b].x, pts[b].y);
      ctx.stroke();
    }
  }

  if (pointsToggle.checked) {
    ctx.fillStyle = "#ff6ad5";
    for (const p of pts) {
      if (p.score < CONFIDENT) continue;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}
