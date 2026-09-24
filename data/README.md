# VisionBridge data workspace

The active product trains on an Indian Sign Language alphabet/fingerspelling dataset.

Recommended flow:

1. Obtain an ISL A-Z image dataset that can be legally used for the project.
2. Prepare the images with the letter-landmark extractor:
   `backend/scripts/prepare_letter_dataset.py`
3. Review class coverage and landmark extraction failures.
4. Train the configurable letter base model with:
   `backend/app/training/letter_base.py`
5. Validate the held-out test split.
6. Install the validated checkpoint at:
   `backend/app/models/weights/letter_base_model.pt`

Prepared data is expected to contain:

```text
visionbridge_letter_data/
├── train.npz
├── val.npz
├── test.npz
└── labels.json
```

The repository should not contain the raw dataset. Keep large datasets and generated training outputs in Colab or the local training workspace.

Signer calibration happens later from a small number of real examples and does not require a separate offline adapter-training job.


## Current training preparation

The active VisionBridge pipeline uses the RealSign ISL A-Z image dataset. The dataset archive is stored with Git LFS, so the Colab notebook downloads the real archive from the Git LFS media endpoint rather than the ordinary raw GitHub file URL. Landmark extraction uses MediaPipe Tasks Hand Landmarker and produces the normalized 126D two-hand representation expected by the base model.


Prepared outputs now also include reproducibility manifests:

```text
visionbridge_letter_data/
├── train_manifest.jsonl
├── val_manifest.jsonl
├── test_manifest.jsonl
└── duplicate_report.json
```

The manifests record source paths, class labels, source split, and exact image SHA-256 values. Exact duplicates are kept together during the generated train/validation split; duplicates involving the source test split are reported rather than removed or relabeled.
