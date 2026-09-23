from pathlib import Path

import numpy as np
import pytest

from app.services import letter_fewshot


def _hand(seed: float) -> list[float]:
    rng = np.random.default_rng(int(seed * 1000))
    points = rng.normal(0, 0.02, (21, 3)).astype(np.float32)
    points[0] = 0
    return points.reshape(-1).tolist()


def _pair(left_seed: float, right_seed: float) -> list[float]:
    return _hand(left_seed) + _hand(right_seed)


def test_normalize_hand_pair_has_stable_contract():
    values = _pair(1, 2)
    result = letter_fewshot.normalize_hand_pair(values)
    assert result.shape == (126,)
    assert np.isfinite(result).all()


def test_fit_and_predict_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(letter_fewshot.settings, "ADAPTER_WEIGHTS_DIR", str(tmp_path))
    samples = [
        ("A", _pair(1, 1.1)),
        ("A", _pair(1.01, 1.11)),
        ("B", _pair(2, 2.1)),
        ("B", _pair(2.01, 2.11)),
    ]
    fitted = letter_fewshot.fit_prototype_adapter(samples)
    path = letter_fewshot.save_prototype_adapter(fitted["payload"])
    loaded = letter_fewshot.load_prototype_adapter(path)
    letter, confidence, scores = letter_fewshot.predict_letter(loaded, samples[0][1])
    assert letter == "A"
    assert confidence > 0.5
    assert scores[0][0] == "A"


def test_requires_multiple_letters():
    with pytest.raises(ValueError, match="at least two"):
        letter_fewshot.fit_prototype_adapter([("A", _pair(1, 1.1))])


def test_rejects_wrong_dimension():
    with pytest.raises(ValueError, match="126"):
        letter_fewshot.normalize_hand_pair([0.0] * 125)
