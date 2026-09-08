import numpy as np
import scipy.io as sio
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset

# LSP joint order (0-indexed):
JOINT_NAMES = [
    "r_ankle", "r_knee", "r_hip", "l_hip", "l_knee", "l_ankle",
    "r_wrist", "r_elbow", "r_shoulder", "l_shoulder", "l_elbow", "l_wrist",
    "neck", "head_top",
]
NUM_JOINTS = 14

# Pairs to swap when horizontally flipping the image (left/right joints).
FLIP_PAIRS = [(0, 5), (1, 4), (2, 3), (6, 11), (7, 10), (8, 9)]

SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),           # legs
    (2, 8), (3, 9),                                     # hips to shoulders
    (8, 9),                                             # shoulders
    (8, 7), (7, 6), (9, 10), (10, 11),                 # arms
    (8, 12), (9, 12), (12, 13),                        # neck/head
]

IMG_SIZE = 256
HEATMAP_SIZE = 64
HEATMAP_STRIDE = IMG_SIZE // HEATMAP_SIZE
SIGMA = 2.0


def _gaussian_heatmap(size, center, sigma):
    x = np.arange(0, size, 1, dtype=np.float32)
    y = x[:, None]
    x0, y0 = center
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))


class LSPDataset(Dataset):
    def __init__(self, root, split="train", val_fraction=0.1, seed=0, augment=None):
        root = Path(root)
        self.images_dir = root / "images"
        mat = sio.loadmat(str(root / "joints.mat"))["joints"]  # (3, 14, N)
        n = mat.shape[2]
        joints = mat.transpose(2, 1, 0)  # (N, 14, 3): x, y, flag
        # LSP convention: flag == 0 means visible.
        vis = (joints[:, :, 2] == 0).astype(np.float32)
        coords = joints[:, :, :2].astype(np.float32)

        rng = np.random.RandomState(seed)
        indices = rng.permutation(n)
        n_val = max(1, int(n * val_fraction))
        val_idx = set(indices[:n_val].tolist())

        self.samples = []
        for i in range(n):
            is_val = i in val_idx
            if split == "train" and is_val:
                continue
            if split == "val" and not is_val:
                continue
            self.samples.append(i)

        self.coords = coords
        self.vis = vis
        self.augment = augment if augment is not None else (split == "train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        i = self.samples[idx]
        img_path = self.images_dir / f"im{i + 1:04d}.jpg"
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        coords = self.coords[i].copy()  # (14, 2) in original pixel space
        vis = self.vis[i].copy()

        flip = self.augment and np.random.rand() < 0.5
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            coords[:, 0] = w - 1 - coords[:, 0]
            for a, b in FLIP_PAIRS:
                coords[[a, b]] = coords[[b, a]]
                vis[[a, b]] = vis[[b, a]]

        # Resize (with distortion) straight to IMG_SIZE -- LSP images are
        # already person-centered crops, so this keeps preprocessing simple.
        sx = IMG_SIZE / w
        sy = IMG_SIZE / h
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        coords[:, 0] *= sx
        coords[:, 1] *= sy

        if self.augment:
            # mild color jitter
            arr = np.asarray(img).astype(np.float32)
            arr *= np.random.uniform(0.8, 1.2)
            arr += np.random.uniform(-15, 15)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        img_arr = np.asarray(img).astype(np.float32) / 255.0
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
        }
