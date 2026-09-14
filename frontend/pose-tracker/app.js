import {
  IMG_SIZE,
  NUM_JOINTS,
  decodeHeatmaps,
  computeCenterCrop,
  imageDataToCHW,
  mapKeypointsToFrame,
  deriveNeckAndChest,
} from "./pose-math.js";

// Indices 0-16 are the model's raw COCO outputs. 17 (neck) and 18 (chest)
// aren't predicted by the model -- COCO has no such keypoints -- they're
// derived client-side as the shoulder midpoint and the neck/hip midpoint,
// same trick OpenPose's COCO+neck variant uses.
const NECK = 17;
const CHEST = 18;
const SKELETON = [
  [15, 13], [13, 11], [16, 14], [14, 12],       // legs
  [11, CHEST], [12, CHEST],                      // hips to chest
  [CHEST, NECK],                                 // chest to neck
  [NECK, 5], [NECK, 6],                          // neck to shoulders
  [5, 7], [7, 9], [6, 8], [8, 10],               // arms to hands (wrists)
  [NECK, 0], [0, 1], [0, 2], [1, 3], [2, 4],     // neck to head/face
];
const FACE_JOINTS = new Set([0, 1, 2, 3, 4]); // nose, eyes, ears -- drawn smaller

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

statParams.textContent = "15,376,721";
statPck.textContent = "0.93";

let session = null;
let running = false;
let rafId = null;

const fpsWindow = [];

// Offscreen canvas used to crop+resize the video frame to the model's
// 256x256 square input (center-cropped to match the person-centered square
// crops the model was trained on).
const inputCanvas = document.createElement("canvas");
inputCanvas.width = IMG_SIZE;
inputCanvas.height = IMG_SIZE;
const inputCtx = inputCanvas.getContext("2d", { willReadFrequently: true });

let cropSide = 0;
let cropOffsetX = 0;
let cropOffsetY = 0;

applyMirror();
mirrorToggle.addEventListener("change", applyMirror);

const breakdownToggle = document.getElementById("breakdown-toggle");
const breakdownPanel = document.getElementById("breakdown-panel");
const breakdownCaret = document.getElementById("breakdown-caret");
breakdownToggle.addEventListener("click", () => {
  const open = breakdownPanel.classList.toggle("hidden") === false;
  breakdownCaret.textContent = open ? "▾" : "▸";
});

function applyMirror() {
  const mirrored = mirrorToggle.checked;
  video.classList.toggle("mirrored", mirrored);
  canvas.classList.toggle("mirrored", mirrored);
}

async function loadModel() {
  engineBadge.textContent = "loading model…";
  engineBadge.className = "badge";

  // Multi-threaded WASM (numThreads > 1) spawns a pool of Web Workers and
  // waits on all of them during InferenceSession.create() -- on some
  // browser/environment combinations that pool init silently deadlocks
  // with zero console output (no error, no rejection, just a promise that
  // never settles). Single-threaded avoids that whole code path: the
  // model still runs entirely in WASM/SIMD on the main thread, just
  // without the worker pool. This one small model doesn't need the
  // parallelism enough to be worth the risk.
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

  const crop = computeCenterCrop(video.videoWidth, video.videoHeight);
  cropSide = crop.side;
  cropOffsetX = crop.offsetX;
  cropOffsetY = crop.offsetY;
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
  inputCtx.drawImage(
    video,
    cropOffsetX, cropOffsetY, cropSide, cropSide,
    0, 0, IMG_SIZE, IMG_SIZE,
  );
  const { data } = inputCtx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
  const chw = imageDataToCHW(data);
  return new ort.Tensor("float32", chw, [1, 3, IMG_SIZE, IMG_SIZE]);
}

async function loop() {
  if (!running) return;

  const frameStart = performance.now();
  const inputTensor = preprocess();

  const inferStart = performance.now();
  const results = await session.run({ input: inputTensor });
  const inferMs = performance.now() - inferStart;

  const keypoints = decodeHeatmaps(results.heatmaps.data);

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
  // Map from normalized crop-space back into the full video frame (the
  // canvas covers the whole frame, but the model only saw the center
  // square crop).
  const pts = mapKeypointsToFrame(keypoints, cropOffsetX, cropOffsetY, cropSide);

  // Derived neck (17) and chest (18) -- see the SKELETON comment above.
  const { neck, chest } = deriveNeckAndChest(pts, 5, 6, 11, 12);
  pts[NECK] = neck;
  pts[CHEST] = chest;

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
    pts.forEach((p, j) => {
      if (p.score < CONFIDENT) return;
      const r = FACE_JOINTS.has(j) ? 2.5 : 4;
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fill();
    });
  }
}
