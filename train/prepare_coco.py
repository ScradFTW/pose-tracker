"""Filter COCO person-keypoints annotations down to a usable instance list
and bulk-download only the images we actually need (not the full 18GB zip).
"""
import json
import os
import random
import concurrent.futures as cf
from pathlib import Path

import requests

ROOT = Path("data/coco")
MAX_TRAIN_IMAGES = 45000
MIN_KEYPOINTS = 5
MIN_AREA = 32 * 32


def load_instances(split, max_images=None):
    with open(ROOT / f"annotations/person_keypoints_{split}2017.json") as f:
        d = json.load(f)

    images_by_id = {img["id"]: img for img in d["images"]}
    instances = []
    for ann in d["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        if ann["num_keypoints"] < MIN_KEYPOINTS:
            continue
        if ann["area"] < MIN_AREA:
            continue
        img = images_by_id[ann["image_id"]]
        instances.append({
            "image_id": ann["image_id"],
            "file_name": img["file_name"],
            "coco_url": img["coco_url"],
            "bbox": ann["bbox"],
            "keypoints": ann["keypoints"],
        })

    if max_images is not None:
        random.Random(0).shuffle(instances)
        image_ids_kept = set()
        kept = []
        for inst in instances:
            if len(image_ids_kept) >= max_images and inst["image_id"] not in image_ids_kept:
                continue
            image_ids_kept.add(inst["image_id"])
            kept.append(inst)
        instances = kept

    return instances


def download_images(instances, split):
    out_dir = ROOT / f"{split}2017"
    out_dir.mkdir(parents=True, exist_ok=True)
    needed = {inst["file_name"]: inst["coco_url"] for inst in instances}
    todo = [(name, url) for name, url in needed.items() if not (out_dir / name).exists()]
    print(f"{split}: {len(needed)} unique images needed, {len(todo)} to download")

    session = requests.Session()

    def fetch(item):
        name, url = item
        try:
            r = session.get(url, timeout=20)
            r.raise_for_status()
            with open(out_dir / name, "wb") as f:
                f.write(r.content)
            return True
        except Exception as e:
            print("failed", name, e)
            return False

    done = 0
    with cf.ThreadPoolExecutor(max_workers=32) as ex:
        for ok in ex.map(fetch, todo):
            done += 1
            if done % 2000 == 0:
                print(f"  {done}/{len(todo)}")


def main():
    train_instances = load_instances("train", max_images=MAX_TRAIN_IMAGES)
    val_instances = load_instances("val", max_images=None)
    print(f"train instances: {len(train_instances)}  val instances: {len(val_instances)}")

    download_images(train_instances, "train")
    download_images(val_instances, "val")

    with open(ROOT / "train_instances.json", "w") as f:
        json.dump(train_instances, f)
    with open(ROOT / "val_instances.json", "w") as f:
        json.dump(val_instances, f)
    print("wrote train_instances.json / val_instances.json")


if __name__ == "__main__":
    main()
