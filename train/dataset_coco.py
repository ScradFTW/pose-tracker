import json
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset

JOINT_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
]
NUM_JOINTS = 17

FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

# 0-indexed version of COCO's official skeleton connectivity.
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]

IMG_SIZE = 256
HEATMAP_SIZE = 64
HEATMAP_STRIDE = IMG_SIZE // HEATMAP_SIZE
SIGMA = 2.0
BOX_PADDING = 1.3


def _gaussian_heatmap(size, center, sigma):
    x = np.arange(0, size, 1, dtype=np.float32)
    y = x[:, None]
    x0, y0 = center
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))


class CocoPoseDataset(Dataset):
    def __init__(self, root, split="train", augment=None):
        root = Path(root)
        with open(root / f"{split}_instances.json") as f:
            self.instances = json.load(f)
        self.images_dir = root / f"{split}2017"
        self.augment = augment if augment is not None else (split == "train")

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        inst = self.instances[idx]
        img_path = self.images_dir / inst["file_name"]
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        kp = np.array(inst["keypoints"], dtype=np.float32).reshape(NUM_JOINTS, 3)
        coords = kp[:, :2].copy()
        vis = (kp[:, 2] > 0).astype(np.float32)

        x, y, w, h = inst["bbox"]
        cx, cy = x + w / 2, y + h / 2
        box_size = max(w, h) * BOX_PADDING

        if self.augment:
            box_size *= np.random.uniform(0.85, 1.15)
            cx += np.random.uniform(-0.1, 0.1) * box_size
            cy += np.random.uniform(-0.1, 0.1) * box_size

        half = box_size / 2
        left, top = cx - half, cy - half

        flip = self.augment and np.random.rand() < 0.5
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            coords[:, 0] = img_w - 1 - coords[:, 0]
            left = img_w - 1 - (left + box_size)
            for a, b in FLIP_PAIRS:
                coords[[a, b]] = coords[[b, a]]
                vis[[a, b]] = vis[[b, a]]

        # Crop (may extend past image bounds; pad with black).
        crop = Image.new("RGB", (int(box_size), int(box_size)))
        src_box = (int(left), int(top), int(left + box_size), int(top + box_size))
        crop.paste(img.crop((max(src_box[0], 0), max(src_box[1], 0),
                              min(src_box[2], img_w), min(src_box[3], img_h))),
                   (max(-src_box[0], 0), max(-src_box[1], 0)))

        coords[:, 0] = coords[:, 0] - left
        coords[:, 1] = coords[:, 1] - top

        scale = IMG_SIZE / box_size
        crop = crop.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        coords *= scale

        out_of_frame = (coords[:, 0] < 0) | (coords[:, 0] >= IMG_SIZE) | \
                        (coords[:, 1] < 0) | (coords[:, 1] >= IMG_SIZE)
        vis[out_of_frame] = 0

        if self.augment:
            arr = np.asarray(crop).astype(np.float32)
            arr *= np.random.uniform(0.8, 1.2)
            arr += np.random.uniform(-15, 15)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            crop = Image.fromarray(arr)

        img_arr = np.asarray(crop).astype(np.float32) / 255.0
        img_arr = (img_arr - 0.5) / 0.5
        img_tensor = torch.from_numpy(img_arr.transpose(2, 0, 1)).float()

        heat_coords = coords / HEATMAP_STRIDE
        heatmaps = np.zeros((NUM_JOINTS, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
        for j in range(NUM_JOINTS):
            x0, y0 = heat_coords[j]
            if vis[j] < 0.5 or not (0 <= x0 < HEATMAP_SIZE and 0 <= y0 < HEATMAP_SIZE):
                continue
            heatmaps[j] = _gaussian_heatmap(HEATMAP_SIZE, (x0, y0), SIGMA)

        return {
            "image": img_tensor,
            "heatmaps": torch.from_numpy(heatmaps),
            "vis": torch.from_numpy(vis),
            "coords": torch.from_numpy(coords),
            "box_size": torch.tensor(float(box_size) * scale),
        }
