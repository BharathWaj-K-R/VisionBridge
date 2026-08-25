from pathlib import Path

import numpy as np
import pytest
import torch

from app.training.isltranslate import ISLTranslateKeypointDataset, SimpleCharTokenizer, ctc_min_input_length


def _write_example(root: Path, uid: str, text: str, frames: int = 4) -> None:
    np.save(root / "pose" / f"{uid}.npy", np.zeros((frames, 132), dtype=np.float32))
    np.save(root / "face" / f"{uid}.npy", np.zeros((frames, 1404), dtype=np.float32))
    (root / "ISLTranslate.csv").write_text(f"uid,text\n{uid},{text}\n", encoding="utf-8")


def test_tokenizer_rejects_unknown_characters() -> None:
    tokenizer = SimpleCharTokenizer()
    with pytest.raises(ValueError, match="Unsupported target characters"):
        tokenizer.encode("hello@world")


def test_ctc_min_input_length_accounts_for_adjacent_repeats() -> None:
    tokenizer = SimpleCharTokenizer()
    labels = torch.tensor(tokenizer.encode("hello"), dtype=torch.long)
    # 'll' requires an intervening CTC blank: 5 target chars + 1 repeat = 6 frames.
    assert ctc_min_input_length(labels) == 6


def test_dataset_rejects_target_that_requires_more_ctc_frames(tmp_path: Path) -> None:
    root = tmp_path / "isltranslate"
    (root / "pose").mkdir(parents=True)
    (root / "face").mkdir(parents=True)
    _write_example(root, "repeat", "ll", frames=2)
    dataset = ISLTranslateKeypointDataset(root)

    with pytest.raises(ValueError, match="CTC target cannot align"):
        from app.training.isltranslate import collate_ctc_batch
        collate_ctc_batch([dataset[0]])
