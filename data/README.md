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

The RealSign training archive is not stored in the VisionBridge repository. The Colab notebook downloads the 656 MB archive directly from the public RealSign source and verifies its exact size and SHA-256 before extraction. Generated landmark arrays and training outputs remain in the training workspace.

Signer calibration happens later from a small number of real examples and does not require a separate offline adapter-training job.


## Current training preparation

The active VisionBridge pipeline uses the RealSign ISL A-Z image dataset. The Colab notebook downloads the archive directly from the RealSign source media endpoint rather than from a VisionBridge Git-LFS pointer. Landmark extraction uses MediaPipe Tasks Hand Landmarker and produces the normalized 126D two-hand representation expected by the base model.


Prepared outputs now also include reproducibility manifests:

```text
visionbridge_letter_data/
├── train_manifest.jsonl
├── val_manifest.jsonl
├── test_manifest.jsonl
└── duplicate_report.json
```

The manifests record source paths, class labels, source split, and exact image SHA-256 values. Exact duplicates are kept together during the generated train/validation split; duplicates involving the source test split are reported rather than removed or relabeled.


## RealSign source training input

The training archive is downloaded directly from the public RealSign repository:

~~~text
source repository:
https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset

source URL:
https://media.githubusercontent.com/media/RealSign62/RealSign-Indian-Sign-Language-Dataset/main/Dataset.zip

expected size:
656689688 bytes

expected SHA-256:
008cae248e346b8c31fbbea057fcc3f69c6909d29a88e6bb1fb0369f528de2b5
~~~

The notebook verifies the size, SHA-256, and ZIP structure before extraction. The VisionBridge repository does not track the dataset archive and does not require Git LFS.
