import numpy as np

import evaluate_full_imagecas_test_mps as metrics


def test_binary_metrics_perfect():
    label = np.zeros((8, 8, 8), dtype=bool)
    label[2:4, 2:4, 2:4] = True
    out = metrics.binary_metrics(label, label)
    assert out["dice"] == 1.0
    assert out["precision"] == 1.0
    assert out["recall"] == 1.0


def test_soft_dice_range():
    label = np.zeros((4, 4, 4), dtype=np.uint8)
    probs = np.zeros((4, 4, 4), dtype=np.float32)
    label[1, 1, 1] = 1
    probs[1, 1, 1] = 0.8
    assert 0.0 <= metrics.soft_dice(probs, label) <= 1.0
