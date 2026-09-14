// Pure, DOM-free math shared by the pose-tracker frontend (app.js) and its
// unit tests (pose-math.test.js). No `document`/`navigator`/canvas access,
// no imports, no side effects at module scope -- every export here is a
// plain function of its arguments so it can be unit tested with node:test
// and nothing else.

export const IMG_SIZE = 256;
export const HEATMAP_SIZE = 64;
export const HEATMAP_STRIDE = IMG_SIZE / HEATMAP_SIZE;
export const NUM_JOINTS = 17;

/**
 * Finds the argmax pixel per joint heatmap and converts it to a point
 * normalized to [0, 1] within the square crop the model saw.
 *
 * `data` is flat, one `heatmapSize` x `heatmapSize` plane per joint,
 * joint-major: index = joint * heatmapSize * heatmapSize + y * heatmapSize + x.
 * (This matches ONNX Runtime's `Tensor.data`, but any ArrayLike<number> --
 * a plain Array, a Float32Array, etc. -- works, which is what keeps this
 * testable without onnxruntime-web or a browser.)
 *
 * @param {ArrayLike<number>} data
 * @param {number} [numJoints]
 * @param {number} [heatmapSize]
 * @returns {{x: number, y: number, score: number}[]} one entry per joint,
 *   in joint order; x/y are normalized [0, 1] within the square crop.
 */
export function decodeHeatmaps(data, numJoints = NUM_JOINTS, heatmapSize = HEATMAP_SIZE) {
  const stride = IMG_SIZE / heatmapSize;
  const plane = heatmapSize * heatmapSize;
  const keypoints = [];
  for (let j = 0; j < numJoints; j++) {
    let best = -Infinity;
    let bx = 0;
    let by = 0;
    const offset = j * plane;
    for (let y = 0; y < heatmapSize; y++) {
      for (let x = 0; x < heatmapSize; x++) {
        const v = data[offset + y * heatmapSize + x];
        if (v > best) {
          best = v;
          bx = x;
          by = y;
        }
      }
    }
    keypoints.push({
      x: (bx * stride) / IMG_SIZE,
      y: (by * stride) / IMG_SIZE,
      score: best,
    });
  }
  return keypoints;
}

/**
 * Computes the center-square crop of a video frame: the largest square
 * that fits, centered on both axes. Used to match the person-centered
 * square crops the model was trained on.
 *
 * @param {number} videoWidth
 * @param {number} videoHeight
 * @returns {{side: number, offsetX: number, offsetY: number}}
 */
export function computeCenterCrop(videoWidth, videoHeight) {
  const side = Math.min(videoWidth, videoHeight);
  return {
    side,
    offsetX: (videoWidth - side) / 2,
    offsetY: (videoHeight - side) / 2,
  };
}

/**
 * Converts RGBA image data (as from CanvasRenderingContext2D#getImageData)
 * into a normalized, planar (CHW) Float32Array suitable for feeding the
 * model: three imgSize x imgSize planes in R, G, B order, each pixel
 * scaled from [0, 255] to [-1, 1]. Alpha is ignored.
 *
 * @param {ArrayLike<number>} rgba - flat, row-major, 4 values per pixel.
 * @param {number} [imgSize]
 * @returns {Float32Array} length 3 * imgSize * imgSize.
 */
export function imageDataToCHW(rgba, imgSize = IMG_SIZE) {
  const plane = imgSize * imgSize;
  const chw = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    const r = rgba[i * 4] / 255;
    const g = rgba[i * 4 + 1] / 255;
    const b = rgba[i * 4 + 2] / 255;
    chw[i] = (r - 0.5) / 0.5;
    chw[plane + i] = (g - 0.5) / 0.5;
    chw[plane * 2 + i] = (b - 0.5) / 0.5;
  }
  return chw;
}

/**
 * Maps normalized crop-space keypoints (as produced by decodeHeatmaps)
 * back into full-video-frame pixel coordinates, given the crop this
 * frame was decoded from.
 *
 * @param {{x: number, y: number, score: number}[]} keypoints
 * @param {number} offsetX
 * @param {number} offsetY
 * @param {number} side
 * @returns {{x: number, y: number, score: number}[]}
 */
export function mapKeypointsToFrame(keypoints, offsetX, offsetY, side) {
  return keypoints.map((kp) => ({
    x: offsetX + kp.x * side,
    y: offsetY + kp.y * side,
    score: kp.score,
  }));
}

/**
 * Midpoint of two scored points; score is the min of the two (a joint
 * pair is only as confident as its weaker member).
 *
 * @param {{x: number, y: number, score: number}} a
 * @param {{x: number, y: number, score: number}} b
 */
export function midpoint(a, b) {
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
    score: Math.min(a.score, b.score),
  };
}

/**
 * Derives the "neck" and "chest" pseudo-joints as shoulder/hip midpoints.
 * COCO has no such keypoints -- this is the same trick OpenPose's COCO+neck
 * variant uses: neck = midpoint(leftShoulder, rightShoulder), chest =
 * midpoint(neck, midpoint(leftHip, rightHip)).
 *
 * @param {{x: number, y: number, score: number}[]} pts - frame-space points.
 * @param {number} leftShoulderIdx
 * @param {number} rightShoulderIdx
 * @param {number} leftHipIdx
 * @param {number} rightHipIdx
 * @returns {{neck: {x:number,y:number,score:number}, chest: {x:number,y:number,score:number}}}
 */
export function deriveNeckAndChest(pts, leftShoulderIdx, rightShoulderIdx, leftHipIdx, rightHipIdx) {
  const neck = midpoint(pts[leftShoulderIdx], pts[rightShoulderIdx]);
  const midHip = midpoint(pts[leftHipIdx], pts[rightHipIdx]);
  const chest = midpoint(neck, midHip);
  return { neck, chest };
}
