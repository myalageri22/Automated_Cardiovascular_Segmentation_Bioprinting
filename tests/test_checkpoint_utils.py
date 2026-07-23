from pathlib import Path

import torch

from scripts.checkpoint_utils import load_checkpoint_for_model, normalize_state_dict_keys, test_checkpoint as inspect_checkpoint


def test_normalize_state_dict_keys():
    state = {
        "module.model.conv.weight": torch.ones(1),
        "model.model.0.conv.weight": torch.ones(1),
    }
    normalized = normalize_state_dict_keys(state)
    assert "model.conv.weight" in normalized
    assert "model.0.conv.weight" in normalized


def test_checkpoint_roundtrip(tmp_path: Path):
    model = torch.nn.Conv3d(1, 1, 1)
    path = tmp_path / "ckpt.pt"
    torch.save({"epoch": 3, "model_state_dict": model.state_dict(), "best_metric": 0.5}, path)
    report = inspect_checkpoint(path)
    assert report["readable"]
    assert report["contains_model_keys"]
    assert report["epoch"] == 3
    loaded = torch.nn.Conv3d(1, 1, 1)
    result = load_checkpoint_for_model(loaded, path, strict=True)
    assert result["missing_keys"] == []
