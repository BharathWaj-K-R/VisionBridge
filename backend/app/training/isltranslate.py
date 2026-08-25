"""ISLTranslate dataset loading utilities."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from app.models.base_model import FACE_INPUT_DIM, MAX_SEQUENCE_LENGTH, POSE_INPUT_DIM


@dataclass(frozen=True)
class ISLTranslateExample:
    uid: str
    text: str
    pose_path: Path
    face_path: Path


class SimpleCharTokenizer:
    blank_token = "<blank>"

    def __init__(self, alphabet: str | None = None):
        if alphabet is None:
            alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 .,?!'\"-:;()"
        self.id_to_token = [self.blank_token, *dict.fromkeys(alphabet.lower())]
        self.token_to_id = {token: idx for idx, token in enumerate(self.id_to_token)}

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str) -> list[int]:
        normalized = text.lower()
        unknown = sorted({ch for ch in normalized if ch not in self.token_to_id})
        if unknown:
            raise ValueError(
                f"Unsupported target characters {unknown!r}; extend the tokenizer alphabet before training."
            )
        return [self.token_to_id[ch] for ch in normalized if ch != self.blank_token]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"id_to_token": self.id_to_token}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SimpleCharTokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        tokenizer = cls("")
        tokenizer.id_to_token = payload["id_to_token"]
        if not tokenizer.id_to_token or tokenizer.id_to_token[0] != tokenizer.blank_token:
            raise ValueError("Saved tokenizer must reserve token 0 for CTC blank")
        tokenizer.token_to_id = {token: idx for idx, token in enumerate(tokenizer.id_to_token)}
        return tokenizer


class ISLTranslateKeypointDataset(Dataset):
    def __init__(self, root: str | Path, tokenizer: SimpleCharTokenizer | None = None):
        self.root = Path(root)
        self.tokenizer = tokenizer or SimpleCharTokenizer()
        self.examples = self._read_examples()
        if not self.examples:
            raise ValueError(f"No usable ISLTranslate examples found under {self.root}")

    def _read_examples(self) -> list[ISLTranslateExample]:
        metadata_path = self.root / "ISLTranslate.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata CSV: {metadata_path}")

        examples: list[ISLTranslateExample] = []
        seen_uids: set[str] = set()
        with metadata_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = {name.lower(): name for name in (reader.fieldnames or [])}
            uid_field = fieldnames.get("uid") or fieldnames.get("video_uid") or fieldnames.get("id")
            text_field = fieldnames.get("text") or fieldnames.get("translation") or fieldnames.get("english")
            if not uid_field or not text_field:
                raise ValueError("ISLTranslate.csv must include uid and translation/text columns")

            for row_number, row in enumerate(reader, start=2):
                uid = row[uid_field].strip()
                text = row[text_field].strip()
                if not uid or not text:
                    continue
                if uid in seen_uids:
                    raise ValueError(
                        f"Duplicate uid {uid!r} in {metadata_path} at CSV row {row_number}. "
                        "Each training clip must have a globally unique UID."
                    )
                seen_uids.add(uid)
                pose_path = self.root / "pose" / f"{uid}.npy"
                face_path = self.root / "face" / f"{uid}.npy"
                if pose_path.exists() and face_path.exists():
                    examples.append(ISLTranslateExample(uid, text, pose_path, face_path))
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        example = self.examples[index]
        pose = torch.from_numpy(np.load(example.pose_path)).float()
        face = torch.from_numpy(np.load(example.face_path)).float()

        if pose.ndim != 2 or pose.shape[1] != POSE_INPUT_DIM:
            raise ValueError(
                f"Example {example.uid!r} has invalid pose shape {tuple(pose.shape)}; "
                f"expected (frames, {POSE_INPUT_DIM})"
            )
        if face.ndim != 2 or face.shape[1] != FACE_INPUT_DIM:
            raise ValueError(
                f"Example {example.uid!r} has invalid face shape {tuple(face.shape)}; "
                f"expected (frames, {FACE_INPUT_DIM})"
            )
        if pose.shape[0] != face.shape[0] or pose.shape[0] == 0:
            raise ValueError(
                f"Example {example.uid!r} has misaligned/empty streams: "
                f"pose_frames={pose.shape[0]}, face_frames={face.shape[0]}"
            )
        if not torch.isfinite(pose).all() or not torch.isfinite(face).all():
            raise ValueError(f"Example {example.uid!r} contains non-finite keypoint values")

        try:
            labels = torch.tensor(self.tokenizer.encode(example.text), dtype=torch.long)
        except ValueError as exc:
            raise ValueError(f"Example {example.uid!r} has invalid target text: {example.text!r}: {exc}") from exc
        if labels.numel() == 0:
            raise ValueError(f"Example {example.uid!r} has no encodable target text: {example.text!r}")
        return {"uid": example.uid, "pose": pose, "face": face, "labels": labels, "text": example.text}


def ctc_min_input_length(labels: torch.Tensor) -> int:
    if labels.numel() == 0:
        return 0
    repeats = int((labels[1:] == labels[:-1]).sum().item())
    return int(labels.numel()) + repeats


def _downsample_to_max_length(pose: torch.Tensor, face: torch.Tensor, uid: str) -> tuple[torch.Tensor, torch.Tensor]:
    pose_frames = int(pose.shape[0])
    face_frames = int(face.shape[0])
    if pose_frames != face_frames:
        raise ValueError(
            f"pose/face frame count mismatch for uid={uid!r}: {pose_frames} vs {face_frames}"
        )

    if pose_frames <= MAX_SEQUENCE_LENGTH:
        return pose, face

    indices = torch.linspace(0, pose_frames - 1, steps=MAX_SEQUENCE_LENGTH).long()
    return pose[indices], face[indices]


def collate_ctc_batch(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    if not batch:
        raise ValueError("Cannot collate an empty CTC batch")

    sampled_items = []
    for item in batch:
        uid = str(item["uid"])
        pose, face = _downsample_to_max_length(item["pose"], item["face"], uid)  # type: ignore[arg-type]
        labels = item["labels"]  # type: ignore[assignment]
        required_frames = ctc_min_input_length(labels)
        if required_frames > pose.shape[0]:
            raise ValueError(
                f"CTC target cannot align for uid={uid!r}: "
                f"target_length={labels.numel()}, repeated_labels={required_frames - labels.numel()}, "
                f"minimum_required_frames={required_frames}, input_frames={pose.shape[0]}"
            )
        sampled_items.append({**item, "pose": pose, "face": face})

    pose_dim = int(sampled_items[0]["pose"].shape[-1])  # type: ignore[index, union-attr]
    face_dim = int(sampled_items[0]["face"].shape[-1])  # type: ignore[index, union-attr]
    if pose_dim != POSE_INPUT_DIM or face_dim != FACE_INPUT_DIM:
        raise ValueError(
            f"Unexpected batch feature dimensions: pose={pose_dim}, face={face_dim}; "
            f"expected pose={POSE_INPUT_DIM}, face={FACE_INPUT_DIM}"
        )

    max_frames = max(int(item["pose"].shape[0]) for item in sampled_items)  # type: ignore[index, union-attr]
    pose = torch.zeros(len(sampled_items), max_frames, pose_dim)
    face = torch.zeros(len(sampled_items), max_frames, face_dim)
    input_lengths = torch.zeros(len(sampled_items), dtype=torch.long)
    label_chunks = []
    label_lengths = torch.zeros(len(sampled_items), dtype=torch.long)
    uids: list[str] = []
    texts: list[str] = []

    for idx, item in enumerate(sampled_items):
        item_pose = item["pose"]  # type: ignore[assignment]
        item_face = item["face"]  # type: ignore[assignment]
        item_labels = item["labels"]  # type: ignore[assignment]
        frames = int(item_pose.shape[0])
        pose[idx, :frames] = item_pose
        face[idx, :frames] = item_face
        input_lengths[idx] = frames
        label_chunks.append(item_labels)
        label_lengths[idx] = int(item_labels.numel())
        uids.append(str(item["uid"]))
        texts.append(str(item["text"]))

    return {
        "uid": uids,
        "text": texts,
        "pose": pose,
        "face": face,
        "labels": torch.cat(label_chunks),
        "input_lengths": input_lengths,
        "label_lengths": label_lengths,
    }
