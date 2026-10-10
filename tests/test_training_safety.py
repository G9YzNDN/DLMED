"""Regression checks for seeded augmentation and preserving finished experiments."""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset import BrainTumorDataset, get_transforms
from train import main, seed_worker


class TrainingSafetyTests(unittest.TestCase):
    def test_augmentation_seed_controls_sequence(self):
        image = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64).astype(np.uint8)
        mask = (image > 127).astype(np.uint8) * 255
        a, b, c = [get_transforms(True, 64, seed) for seed in (7, 7, 8)]
        different = False
        for _ in range(8):
            x, y, z = [tf(image=image, mask=mask) for tf in (a, b, c)]
            np.testing.assert_array_equal(x["image"], y["image"])
            np.testing.assert_array_equal(x["mask"], y["mask"])
            self.assertTrue(set(np.unique(x["mask"])).issubset({0, 255}))
            different |= not np.array_equal(x["image"], z["image"])
        self.assertTrue(different)

    def test_two_worker_loader_repeats_across_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            image = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64).astype(np.uint8)
            rows = []
            for i in range(8):
                cv2.imwrite(str(root / "images" / f"{i}.png"), np.roll(image, i, axis=0))
                cv2.imwrite(str(root / "masks" / f"{i}.png"), (image > 127).astype(np.uint8) * 255)
                rows.append({"id": str(i), "pid": str(i), "label": i % 3, "split": "train"})
            pd.DataFrame(rows).to_csv(root / "splits.csv", index=False)

            def collect():
                ds = BrainTumorDataset(root, root / "splits.csv", "train", 64, train=True, seed=7)
                loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=2,
                                    persistent_workers=True, worker_init_fn=seed_worker,
                                    generator=torch.Generator().manual_seed(7))
                epochs = []
                for _ in range(2):
                    batches = []
                    for batch in loader:
                        digest = hashlib.sha256(batch["image"].numpy().tobytes() +
                                                batch["mask"].numpy().tobytes()).hexdigest()
                        batches.append((tuple(batch["id"]), digest))
                    epochs.append(batches)
                return epochs

            a, b = collect(), collect()
            self.assertEqual(a, b)
            self.assertNotEqual(a[0], a[1])

    def test_existing_run_is_not_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "finished"
            run.mkdir()
            originals = {"config.json": b'{"seed": 123}', "best.pt": b"saved model"}
            for name, content in originals.items():
                (run / name).write_bytes(content)
            with patch.object(sys, "argv", ["train.py", "--out_dir", tmp, "--name", "finished"]):
                with self.assertRaisesRegex(SystemExit, "already exists"):
                    main()
            self.assertEqual({p.name: p.read_bytes() for p in run.iterdir()}, originals)


if __name__ == "__main__":
    unittest.main()
