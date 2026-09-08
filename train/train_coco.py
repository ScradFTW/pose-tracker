import argparse
import time

import torch
from torch.utils.data import DataLoader

from dataset_coco import CocoPoseDataset, NUM_JOINTS, HEATMAP_STRIDE
from model_resnet import ResNetPoseNet


def heatmaps_to_coords(heatmaps):
    b, j, h, w = heatmaps.shape
    flat = heatmaps.view(b, j, -1)
    idx = flat.argmax(dim=-1)
    ys = (idx // w).float()
    xs = (idx % w).float()
    return torch.stack([xs, ys], dim=-1)


def pck_at(pred_coords, gt_coords, vis, threshold_px):
    dist = torch.norm(pred_coords - gt_coords, dim=-1)
    correct = (dist <= threshold_px.unsqueeze(-1)) & (vis > 0.5)
    total = vis > 0.5
    return correct.sum().item(), total.sum().item()


def run_epoch(model, loader, device, optimizer=None, log_every=200):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    n_batches = 0
    correct, total = 0, 0

    for i, batch in enumerate(loader):
        img = batch["image"].to(device, non_blocking=True)
        heatmaps = batch["heatmaps"].to(device, non_blocking=True)
        vis = batch["vis"].to(device, non_blocking=True)
        coords = batch["coords"].to(device, non_blocking=True)
        box_size = batch["box_size"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            pred = model(img)
            mask = vis.unsqueeze(-1).unsqueeze(-1)
            loss = ((pred - heatmaps) ** 2 * mask).sum() / mask.sum().clamp(min=1)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        pred_px = heatmaps_to_coords(pred.detach()) * HEATMAP_STRIDE
        thresh = 0.2 * box_size
        c, t = pck_at(pred_px, coords, vis, thresh)
        correct += c
        total += t

        if train and (i + 1) % log_every == 0:
            print(f"    step {i + 1}/{len(loader)}  loss={total_loss / n_batches:.5f}  "
                  f"pck={correct / max(total, 1):.3f}", flush=True)

    pck = correct / max(total, 1)
    return total_loss / max(n_batches, 1), pck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/coco")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--out", default="train/pose_net_coco.pt")
    ap.add_argument("--report", default="train/eval_report_coco.txt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    train_ds = CocoPoseDataset(args.data, split="train")
    val_ds = CocoPoseDataset(args.data, split="val", augment=False)
    print(f"train={len(train_ds)} val={len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=8, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, persistent_workers=True)

    model = ResNetPoseNet(NUM_JOINTS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_pck = -1.0
    t0 = time.time()
    history = []
    for epoch in range(1, args.epochs + 1):
        te0 = time.time()
        train_loss, train_pck = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_pck = run_epoch(model, val_loader, device, optimizer=None)
        scheduler.step()

        history.append((epoch, train_loss, train_pck, val_loss, val_pck))
        print(f"epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.5f} train_pck={train_pck:.3f}  "
              f"val_loss={val_loss:.5f} val_pck={val_pck:.3f}  "
              f"({(time.time() - te0):.0f}s)", flush=True)

        if val_pck > best_pck:
            best_pck = val_pck
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_pck": val_pck}, args.out)

    elapsed = time.time() - t0
    print(f"done in {elapsed / 60:.1f} min, best val PCK@0.2={best_pck:.3f}, checkpoint={args.out}", flush=True)

    with open(args.report, "w") as f:
        f.write("Pose-tracker: from-scratch ResNet18-style heatmap CNN on COCO keypoints\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Dataset: COCO 2017 person_keypoints -- {len(train_ds)} train / {len(val_ds)} val instances\n")
        f.write("Note: PCK@0.2 here is normalized by the crop box size (a simplified\n")
        f.write("proxy), not official COCO OKS -- not directly comparable to papers.\n")
        f.write(f"Model: ResNet18-shaped backbone (random init) + 3-stage deconv head\n")
        f.write(f"Model params: {n_params:,}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Training time: {elapsed / 60:.1f} min on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"Best val PCK@0.2 (bbox-normalized): {best_pck:.3f}\n\n")
        f.write("Per-epoch log (epoch, train_loss, train_pck, val_loss, val_pck):\n")
        for row in history:
            f.write(f"{row[0]:3d}  {row[1]:.5f}  {row[2]:.3f}  {row[3]:.5f}  {row[4]:.3f}\n")


if __name__ == "__main__":
    main()
