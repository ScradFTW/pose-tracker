"""Quick visual sanity check: run the current best checkpoint on a grid of
COCO val instances and draw predicted vs ground-truth keypoints."""
import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw

from dataset_coco import CocoPoseDataset, HEATMAP_STRIDE, SKELETON, IMG_SIZE
from model_resnet import ResNetPoseNet, NUM_JOINTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="train/pose_net_coco.pt")
    ap.add_argument("--out", default="/tmp/coco-quality-grid.png")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = ResNetPoseNet(NUM_JOINTS).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded epoch={ckpt['epoch']} val_pck={ckpt['val_pck']:.3f}")

    val_ds = CocoPoseDataset("data/coco", split="val", augment=False)
    cols = 4
    rows = (args.n + cols - 1) // cols
    grid = Image.new("RGB", (IMG_SIZE * cols, IMG_SIZE * rows), "black")

    rng = np.random.RandomState(1)
    idxs = rng.choice(len(val_ds), size=args.n, replace=False)

    with torch.no_grad():
        for i, idx in enumerate(idxs):
            sample = val_ds[int(idx)]
            img_tensor = sample["image"].unsqueeze(0).to(device)
            gt_coords = sample["coords"].numpy()
            heatmaps = model(img_tensor)[0].cpu().numpy()

            pred_coords = []
            for j in range(NUM_JOINTS):
                yy, xx = np.unravel_index(np.argmax(heatmaps[j]), heatmaps[j].shape)
                pred_coords.append((xx * HEATMAP_STRIDE, yy * HEATMAP_STRIDE))

            img_arr = ((sample["image"].numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(img_arr)
            draw = ImageDraw.Draw(img)
            for a, b in SKELETON:
                draw.line([pred_coords[a], pred_coords[b]], fill=(79, 209, 197), width=2)
            for x, y in pred_coords:
                draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 106, 213))
            for x, y in gt_coords:
                draw.ellipse([x - 2, y - 2, x + 2, y + 2], outline=(255, 255, 0))

            grid.paste(img, ((i % cols) * IMG_SIZE, (i // cols) * IMG_SIZE))

    grid.save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
