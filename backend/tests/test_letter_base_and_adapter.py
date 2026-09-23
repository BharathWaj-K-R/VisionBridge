import numpy as np
import pytest
import torch

from app.models.letter_model import VisionBridgeLetterBaseModel, build_browser_payload, save_checkpoint, load_checkpoint
from app.services import letter_fewshot


def _pair(seed):
    rng = np.random.default_rng(seed)
    points = rng.normal(0, 0.02, (21, 3)).astype(np.float32)
    points[0] = 0
    return points.reshape(-1).tolist() + points.reshape(-1).tolist()


def test_base_model_contract():
    model = VisionBridgeLetterBaseModel()
    x = torch.randn(4, 126)
    logits = model(x)
    emb = model.embed(x)
    assert logits.shape == (4, 26)
    assert emb.shape == (4, 64)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(emb).all()


def test_base_model_scales_from_config():
    model = VisionBridgeLetterBaseModel(
        input_dim=126,
        hidden_dim=256,
        embedding_dim=96,
        labels=("A", "B", "C", "D", "E", "F"),
    )
    x = torch.randn(3, 126)
    assert model(x).shape == (3, 6)
    assert model.embed(x).shape == (3, 96)


def test_checkpoint_round_trip_preserves_dynamic_configuration(tmp_path):
    path = tmp_path / "base.pt"
    model = VisionBridgeLetterBaseModel(
        hidden_dim=192,
        embedding_dim=80,
        labels=("A", "B", "C"),
        dropout=0.2,
    )
    save_checkpoint(model, path)
    loaded = load_checkpoint(path)

    assert loaded.input_dim == 126
    assert loaded.hidden_dim == 192
    assert loaded.embedding_dim == 80
    assert loaded.labels == ("A", "B", "C")
    assert loaded.output_head.out_features == 3
    assert loaded.dropout == pytest.approx(0.2)


def test_few_shot_adapter_tracks_current_base_version(tmp_path, monkeypatch):
    model = VisionBridgeLetterBaseModel(embedding_dim=80)
    base = tmp_path / "base.pt"
    save_checkpoint(model, base)
    monkeypatch.setattr(letter_fewshot.settings, "LETTER_BASE_MODEL_PATH", str(base))
    monkeypatch.setattr(letter_fewshot.settings, "ADAPTER_WEIGHTS_DIR", str(tmp_path / "adapters"))

    fitted = letter_fewshot.fit_prototype_adapter(
        model,
        [("A", _pair(1)), ("A", _pair(1)), ("B", _pair(2)), ("B", _pair(2))],
    )
    adapter_path = letter_fewshot.save_prototype_adapter(fitted["payload"])
    loaded = letter_fewshot.load_prototype_adapter(adapter_path, base)
    pred, conf, scores = letter_fewshot.predict_letter(model, loaded, _pair(1))

    assert loaded["method"] == "dynamic-base-embedding-prototype"
    assert loaded["embedding_dim"] == 80
    assert loaded["base_model_version"] != ""
    assert pred == "A"
    assert 0 < conf <= 1
    assert scores[0][0] == "A"


def test_adapter_auto_refreshes_after_base_change(tmp_path, monkeypatch):
    model = VisionBridgeLetterBaseModel(embedding_dim=80)
    a = tmp_path / "a.pt"
    b = tmp_path / "b.pt"
    save_checkpoint(model, a)
    changed = VisionBridgeLetterBaseModel(hidden_dim=256, embedding_dim=96)
    save_checkpoint(changed, b)

    monkeypatch.setattr(letter_fewshot.settings, "LETTER_BASE_MODEL_PATH", str(a))
    monkeypatch.setattr(letter_fewshot.settings, "ADAPTER_WEIGHTS_DIR", str(tmp_path / "adapters"))

    fitted = letter_fewshot.fit_prototype_adapter(model, [("A", _pair(1)), ("B", _pair(2))])
    path = letter_fewshot.save_prototype_adapter(fitted["payload"])

    refreshed = letter_fewshot.load_prototype_adapter(path, b)

    assert refreshed["base_model_version"] != ""
    assert refreshed["embedding_dim"] == 96
    assert refreshed["shots"] == {"A": 1, "B": 1}
    assert all(len(values) == 96 for values in refreshed["prototypes"].values())

def test_browser_payload_matches_checkpoint(tmp_path):
    path = tmp_path / "base.pt"
    model = VisionBridgeLetterBaseModel(hidden_dim=160, embedding_dim=72, labels=("A", "B", "C"))
    save_checkpoint(model, path)

    payload = build_browser_payload(model, "sha")
    assert payload["hidden_dim"] == 160
    assert payload["embedding_dim"] == 72
    assert payload["num_classes"] == 3
    assert len(payload["layers"]["hidden"]["weight"]) == 160
    assert len(payload["layers"]["head"]["weight"]) == 3


def test_degenerate_input_rejected():
    with pytest.raises(ValueError, match="No visible"):
        letter_fewshot.normalize_hand_pair([0.0] * 126)
