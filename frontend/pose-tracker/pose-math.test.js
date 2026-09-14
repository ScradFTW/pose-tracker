// Unit tests for pose-math.js, the DOM-free math extracted from app.js.
// Run with `node --test` from this directory (or `npm test`) -- no
// dependencies, no bundler, no browser needed.

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
  IMG_SIZE,
  HEATMAP_SIZE,
  HEATMAP_STRIDE,
  NUM_JOINTS,
  decodeHeatmaps,
  computeCenterCrop,
  imageDataToCHW,
  mapKeypointsToFrame,
  midpoint,
  deriveNeckAndChest,
} from "./pose-math.js";

describe("decodeHeatmaps", () => {
  it("finds the argmax pixel of a single joint and normalizes it", () => {
    // 4x4 heatmap, one joint, peak at (x=3, y=2).
    const size = 4;
    const data = new Float32Array(size * size).fill(0);
    data[2 * size + 3] = 0.9; // y=2, x=3
    const [kp] = decodeHeatmaps(data, 1, size);

    const stride = IMG_SIZE / size; // 64
    assert.equal(kp.score, Math.fround(0.9)); // data is a Float32Array
    assert.equal(kp.x, (3 * stride) / IMG_SIZE);
    assert.equal(kp.y, (2 * stride) / IMG_SIZE);
  });

  it("decodes each joint independently from its own plane", () => {
    const size = 2;
    const plane = size * size;
    // Joint 0 peaks at (0,0), joint 1 peaks at (1,1).
    const data = new Float32Array(plane * 2).fill(-1);
    data[0 * plane + (0 * size + 0)] = 5; // joint 0, x=0,y=0
    data[1 * plane + (1 * size + 1)] = 7; // joint 1, x=1,y=1

    const keypoints = decodeHeatmaps(data, 2, size);
    assert.equal(keypoints.length, 2);

    const stride = IMG_SIZE / size;
    assert.deepEqual(keypoints[0], { x: 0, y: 0, score: 5 });
    assert.deepEqual(keypoints[1], {
      x: (1 * stride) / IMG_SIZE,
      y: (1 * stride) / IMG_SIZE,
      score: 7,
    });
  });

  it("picks the first pixel (x=0,y=0) when every value is equal", () => {
    // Strict `v > best` means ties never displace the first max found.
    const size = 3;
    const data = new Float32Array(size * size).fill(0.42);
    const [kp] = decodeHeatmaps(data, 1, size);
    assert.deepEqual(kp, { x: 0, y: 0, score: Math.fround(0.42) }); // data is a Float32Array
  });

  it("handles all-negative heatmaps (still finds the least-negative peak)", () => {
    const size = 2;
    const data = new Float32Array([-5, -1, -3, -2]); // best is -1 at x=1,y=0
    const [kp] = decodeHeatmaps(data, 1, size);
    const stride = IMG_SIZE / size;
    assert.equal(kp.score, -1);
    assert.equal(kp.x, (1 * stride) / IMG_SIZE);
    assert.equal(kp.y, 0);
  });

  it("normalizes a peak at the last pixel of the grid to just under 1.0", () => {
    const size = 8;
    const data = new Float32Array(size * size).fill(0);
    data[(size - 1) * size + (size - 1)] = 1; // bottom-right corner
    const [kp] = decodeHeatmaps(data, 1, size);
    const stride = IMG_SIZE / size;
    const expected = ((size - 1) * stride) / IMG_SIZE;
    assert.equal(kp.x, expected);
    assert.equal(kp.y, expected);
    assert.ok(kp.x < 1 && kp.y < 1);
  });

  it("works at the real model shape (NUM_JOINTS x HEATMAP_SIZE^2) with default args", () => {
    const plane = HEATMAP_SIZE * HEATMAP_SIZE;
    const data = new Float32Array(NUM_JOINTS * plane).fill(0);
    // Give joint 0 a peak at (10, 20) and joint 16 a peak at (63, 0).
    data[0 * plane + 20 * HEATMAP_SIZE + 10] = 1;
    data[16 * plane + 0 * HEATMAP_SIZE + 63] = 1;

    const keypoints = decodeHeatmaps(data);
    assert.equal(keypoints.length, NUM_JOINTS);
    assert.equal(keypoints[0].x, (10 * HEATMAP_STRIDE) / IMG_SIZE);
    assert.equal(keypoints[0].y, (20 * HEATMAP_STRIDE) / IMG_SIZE);
    assert.equal(keypoints[16].x, (63 * HEATMAP_STRIDE) / IMG_SIZE);
    assert.equal(keypoints[16].y, 0);
    // Untouched joints stay at (0,0) with score 0.
    assert.deepEqual(keypoints[1], { x: 0, y: 0, score: 0 });
  });
});

describe("computeCenterCrop", () => {
  it("uses the full frame with no offset when width equals height", () => {
    const crop = computeCenterCrop(480, 480);
    assert.deepEqual(crop, { side: 480, offsetX: 0, offsetY: 0 });
  });

  it("crops the sides for a landscape video (width > height)", () => {
    const crop = computeCenterCrop(640, 480);
    assert.equal(crop.side, 480);
    assert.equal(crop.offsetX, (640 - 480) / 2);
    assert.equal(crop.offsetY, 0);
  });

  it("crops top/bottom for a portrait video (height > width)", () => {
    const crop = computeCenterCrop(480, 640);
    assert.equal(crop.side, 480);
    assert.equal(crop.offsetX, 0);
    assert.equal(crop.offsetY, (640 - 480) / 2);
  });

  it("produces exact fractional offsets for odd width/height differences", () => {
    // 641x480: side=480, offsetX=(641-480)/2=80.5
    const crop = computeCenterCrop(641, 480);
    assert.equal(crop.side, 480);
    assert.equal(crop.offsetX, 80.5);
    assert.equal(crop.offsetY, 0);
  });

  it("handles a zero-sized dimension without throwing", () => {
    const crop = computeCenterCrop(0, 0);
    assert.deepEqual(crop, { side: 0, offsetX: 0, offsetY: 0 });
  });
});

describe("imageDataToCHW", () => {
  it("maps a black pixel to -1 on every channel", () => {
    const rgba = new Uint8ClampedArray([0, 0, 0, 255]);
    const chw = imageDataToCHW(rgba, 1);
    assert.equal(chw.length, 3);
    assert.ok([...chw].every((v) => Math.abs(v - -1) < 1e-6));
  });

  it("maps a white pixel to 1 on every channel", () => {
    const rgba = new Uint8ClampedArray([255, 255, 255, 255]);
    const chw = imageDataToCHW(rgba, 1);
    assert.ok([...chw].every((v) => Math.abs(v - 1) < 1e-6));
  });

  it("normalizes an arbitrary pixel per-channel with the (v/255-0.5)/0.5 formula", () => {
    const rgba = new Uint8ClampedArray([128, 64, 32, 200]);
    const [r, g, b] = imageDataToCHW(rgba, 1);
    assert.ok(Math.abs(r - (128 / 255 - 0.5) / 0.5) < 1e-6);
    assert.ok(Math.abs(g - (64 / 255 - 0.5) / 0.5) < 1e-6);
    assert.ok(Math.abs(b - (32 / 255 - 0.5) / 0.5) < 1e-6);
  });

  it("ignores alpha entirely", () => {
    const opaque = imageDataToCHW(new Uint8ClampedArray([10, 20, 30, 255]), 1);
    const transparent = imageDataToCHW(new Uint8ClampedArray([10, 20, 30, 0]), 1);
    assert.deepEqual([...opaque], [...transparent]);
  });

  it("packs a multi-pixel image planar (all R, then all G, then all B) in row-major order", () => {
    // 2x2 image, four distinct pixels so we can track exactly where each
    // channel value lands.
    const rgba = new Uint8ClampedArray([
      10, 110, 210, 255, // pixel 0 (x=0,y=0)
      20, 120, 220, 255, // pixel 1 (x=1,y=0)
      30, 130, 230, 255, // pixel 2 (x=0,y=1)
      40, 140, 240, 255, // pixel 3 (x=1,y=1)
    ]);
    const chw = imageDataToCHW(rgba, 2);
    assert.equal(chw.length, 3 * 4);

    const norm = (v) => (v / 255 - 0.5) / 0.5;
    // R plane: indices 0-3.
    assert.ok(Math.abs(chw[0] - norm(10)) < 1e-6);
    assert.ok(Math.abs(chw[1] - norm(20)) < 1e-6);
    assert.ok(Math.abs(chw[2] - norm(30)) < 1e-6);
    assert.ok(Math.abs(chw[3] - norm(40)) < 1e-6);
    // G plane: indices 4-7.
    assert.ok(Math.abs(chw[4] - norm(110)) < 1e-6);
    assert.ok(Math.abs(chw[7] - norm(140)) < 1e-6);
    // B plane: indices 8-11.
    assert.ok(Math.abs(chw[8] - norm(210)) < 1e-6);
    assert.ok(Math.abs(chw[11] - norm(240)) < 1e-6);
  });
});

describe("mapKeypointsToFrame", () => {
  it("maps normalized crop-space points into frame pixel coordinates", () => {
    const keypoints = [{ x: 0.5, y: 0.25, score: 0.8 }];
    const pts = mapKeypointsToFrame(keypoints, 100, 10, 200);
    assert.deepEqual(pts, [{ x: 100 + 0.5 * 200, y: 10 + 0.25 * 200, score: 0.8 }]);
  });

  it("is a no-op mapping when offsets are zero and side is 1", () => {
    const keypoints = [{ x: 0.1, y: 0.9, score: 1 }];
    const pts = mapKeypointsToFrame(keypoints, 0, 0, 1);
    assert.deepEqual(pts, keypoints);
  });

  it("preserves order and length across multiple keypoints", () => {
    const keypoints = [
      { x: 0, y: 0, score: 0 },
      { x: 1, y: 1, score: 1 },
      { x: 0.5, y: 0.5, score: 0.5 },
    ];
    const pts = mapKeypointsToFrame(keypoints, 5, 5, 10);
    assert.equal(pts.length, 3);
    assert.deepEqual(pts[1], { x: 15, y: 15, score: 1 });
  });
});

describe("midpoint", () => {
  it("averages x/y and takes the minimum score", () => {
    const a = { x: 0, y: 0, score: 0.9 };
    const b = { x: 10, y: 20, score: 0.3 };
    assert.deepEqual(midpoint(a, b), { x: 5, y: 10, score: 0.3 });
  });

  it("returns the same point (with its score) when both inputs are identical", () => {
    const p = { x: 3, y: 4, score: 0.7 };
    assert.deepEqual(midpoint(p, p), { x: 3, y: 4, score: 0.7 });
  });
});

describe("deriveNeckAndChest", () => {
  it("derives neck as the shoulder midpoint and chest as neck/hip midpoint", () => {
    // Sparse array indexed like the real 19-slot keypoint layout.
    const pts = [];
    pts[5] = { x: 0, y: 0, score: 0.9 }; // left shoulder
    pts[6] = { x: 10, y: 0, score: 0.8 }; // right shoulder
    pts[11] = { x: 0, y: 10, score: 0.7 }; // left hip
    pts[12] = { x: 10, y: 10, score: 0.6 }; // right hip

    const { neck, chest } = deriveNeckAndChest(pts, 5, 6, 11, 12);

    assert.deepEqual(neck, { x: 5, y: 0, score: 0.8 });
    const midHip = { x: 5, y: 10, score: 0.6 };
    assert.deepEqual(chest, {
      x: (neck.x + midHip.x) / 2,
      y: (neck.y + midHip.y) / 2,
      score: Math.min(neck.score, midHip.score),
    });
  });

  it("propagates a low-confidence joint's score through both derived joints", () => {
    const pts = [];
    pts[5] = { x: 0, y: 0, score: 0.02 }; // below the app's 0.05 confidence cutoff
    pts[6] = { x: 10, y: 0, score: 0.9 };
    pts[11] = { x: 0, y: 10, score: 0.9 };
    pts[12] = { x: 10, y: 10, score: 0.9 };

    const { neck, chest } = deriveNeckAndChest(pts, 5, 6, 11, 12);
    assert.equal(neck.score, 0.02);
    assert.equal(chest.score, 0.02);
  });
});
