"""Unit tests for prepare_coco.py's pure annotation-filtering logic.

prepare_coco.py doesn't import torch (unlike almost everything else in
train/), so these run with nothing but the standard library plus
`requests` (already a prepare_coco.py dependency) -- no venv setup, just
`python3 -m unittest discover -s train -p 'test_*.py'`.

Only load_instances (filtering + downsampling) is exercised. download_images
does real network I/O and is intentionally not covered here.
"""
import json
import tempfile
import unittest
from pathlib import Path

import prepare_coco


def _write_annotations(root, split, images, annotations):
    ann_dir = root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    with open(ann_dir / f"person_keypoints_{split}2017.json", "w") as f:
        json.dump({"images": images, "annotations": annotations}, f)


def _valid_annotation(image_id, **overrides):
    ann = {
        "image_id": image_id,
        "iscrowd": 0,
        "num_keypoints": prepare_coco.MIN_KEYPOINTS,
        "area": prepare_coco.MIN_AREA,
        "bbox": [0, 0, 1, 1],
        "keypoints": [],
    }
    ann.update(overrides)
    return ann


class LoadInstancesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        orig_root = prepare_coco.ROOT
        prepare_coco.ROOT = self.root
        self.addCleanup(lambda: setattr(prepare_coco, "ROOT", orig_root))

    def test_drops_crowd_annotations(self):
        images = [{"id": 1, "file_name": "a.jpg", "coco_url": "http://x/a.jpg"}]
        annotations = [_valid_annotation(1, iscrowd=1)]
        _write_annotations(self.root, "train", images, annotations)
        self.assertEqual(prepare_coco.load_instances("train"), [])

    def test_drops_too_few_keypoints(self):
        images = [{"id": 1, "file_name": "a.jpg", "coco_url": "u"}]
        annotations = [_valid_annotation(1, num_keypoints=prepare_coco.MIN_KEYPOINTS - 1)]
        _write_annotations(self.root, "train", images, annotations)
        self.assertEqual(prepare_coco.load_instances("train"), [])

    def test_drops_too_small_area(self):
        images = [{"id": 1, "file_name": "a.jpg", "coco_url": "u"}]
        annotations = [_valid_annotation(1, area=prepare_coco.MIN_AREA - 1)]
        _write_annotations(self.root, "train", images, annotations)
        self.assertEqual(prepare_coco.load_instances("train"), [])

    def test_keeps_instance_exactly_at_the_thresholds(self):
        # num_keypoints == MIN_KEYPOINTS and area == MIN_AREA are both
        # inclusive (the checks are strict `<`), so this must be kept.
        images = [{"id": 7, "file_name": "person.jpg", "coco_url": "http://images/person.jpg"}]
        annotations = [_valid_annotation(7, bbox=[1, 2, 3, 4], keypoints=[1] * 51)]
        _write_annotations(self.root, "train", images, annotations)

        instances = prepare_coco.load_instances("train")
        self.assertEqual(instances, [{
            "image_id": 7,
            "file_name": "person.jpg",
            "coco_url": "http://images/person.jpg",
            "bbox": [1, 2, 3, 4],
            "keypoints": [1] * 51,
        }])

    def test_max_images_caps_distinct_images_but_never_splits_an_image(self):
        # 3 images; image 1 has two valid person instances, images 2 and 3
        # have one each. Capping at max_images=2 must keep at most 2
        # distinct image_ids, but *every* instance belonging to a kept
        # image (never a partial image).
        images = [
            {"id": 1, "file_name": "1.jpg", "coco_url": "u1"},
            {"id": 2, "file_name": "2.jpg", "coco_url": "u2"},
            {"id": 3, "file_name": "3.jpg", "coco_url": "u3"},
        ]
        annotations = [_valid_annotation(1), _valid_annotation(1), _valid_annotation(2), _valid_annotation(3)]
        _write_annotations(self.root, "train", images, annotations)

        instances = prepare_coco.load_instances("train", max_images=2)
        image_ids = {inst["image_id"] for inst in instances}
        self.assertLessEqual(len(image_ids), 2)
        for image_id in image_ids:
            expected = sum(1 for a in annotations if a["image_id"] == image_id)
            actual = sum(1 for inst in instances if inst["image_id"] == image_id)
            self.assertEqual(actual, expected, f"image {image_id} was only partially kept")

    def test_max_images_none_keeps_everything(self):
        images = [{"id": i, "file_name": f"{i}.jpg", "coco_url": f"u{i}"} for i in range(5)]
        annotations = [_valid_annotation(i) for i in range(5)]
        _write_annotations(self.root, "val", images, annotations)

        instances = prepare_coco.load_instances("val", max_images=None)
        self.assertEqual(len(instances), 5)

    def test_deterministic_across_repeated_calls(self):
        # Downsampling shuffles with a fixed seed (random.Random(0)) so the
        # same input must always produce the same kept set.
        images = [{"id": i, "file_name": f"{i}.jpg", "coco_url": f"u{i}"} for i in range(10)]
        annotations = [_valid_annotation(i) for i in range(10)]
        _write_annotations(self.root, "train", images, annotations)

        first = prepare_coco.load_instances("train", max_images=4)
        second = prepare_coco.load_instances("train", max_images=4)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
