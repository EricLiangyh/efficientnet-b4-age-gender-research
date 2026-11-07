import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from age_gender.data import create_manifest, read_manifest, validate_rows
from age_gender.metrics import calculate
from age_gender.model import load_model


class ResearchChecks(unittest.TestCase):
    def test_known_metrics(self):
        result = calculate([10, 20], [12, 16], [0, 1], [0, 0])
        self.assertEqual(result['age_mae_years'], 3)
        self.assertAlmostEqual(result['age_rmse_years'], np.sqrt(10))
        self.assertEqual(result['gender_accuracy'], 0.5)
        self.assertEqual(result['confusion_matrix'], [[1, 0], [1, 0]])

    def test_split_is_stable_and_duplicates_stay_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'UTKFace'
            folder.mkdir()
            for i in range(20):
                Image.new('RGB', (12, 12), (i * 10, 30, 60)).save(folder / f'{i+10}_{i%2}_sample.png')
            (folder / '10_0_duplicate.png').write_bytes((folder / '10_0_sample.png').read_bytes())
            a = create_manifest(root, root / 'a.csv')
            b = create_manifest(root, root / 'b.csv')
            self.assertEqual(a, b)
            self.assertEqual(a, read_manifest(root / 'a.csv'))
            duplicate = [row for row in a if row['age'] == 10]
            self.assertEqual(len({row['split'] for row in duplicate}), 1)
            changed = [dict(row) for row in a]
            changed[0]['path'] = '../outside.png'
            with self.assertRaises(ValueError):
                validate_rows(changed)

    def test_cross_split_duplicate_is_rejected(self):
        row = {'path': 'a.png', 'age': 20, 'gender': 0, 'split': 'train', 'sha256': 'a'*64}
        other = dict(row, path='b.png', split='test')
        with self.assertRaises(ValueError):
            validate_rows([row, other])

    def test_historical_csv_evidence(self):
        root = Path(__file__).resolve().parents[1]
        logs = list((root / 'experiments/logs').glob('*.csv'))
        self.assertEqual(len(logs), 8)
        for log in logs:
            with log.open(encoding='utf-8-sig') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(r['epoch']) for r in rows], list(range(1, len(rows)+1)))
            self.assertTrue(all(0 <= float(r['val_acc']) <= 100 for r in rows))

    def test_checkpoint_load_and_finite_forward(self):
        torch.set_num_threads(2)
        root = Path(__file__).resolve().parents[1]
        model = load_model(root / 'weights/efficientnet_b4_c3_epoch35.pth')
        with torch.inference_mode():
            age, gender = model(torch.zeros(1, 3, 380, 380))
        self.assertEqual(tuple(age.shape), (1, 1))
        self.assertEqual(tuple(gender.shape), (1, 2))
        self.assertTrue(torch.isfinite(age).all() and torch.isfinite(gender).all())


if __name__ == '__main__':
    unittest.main()
