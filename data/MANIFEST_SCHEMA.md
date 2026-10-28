# Dataset manifest schema v2

Manifest v2 is UTF-8 JSON Lines: exactly one JSON object per sample. Keep raw images and generated arrays outside Git. The manifest contains provenance and capture metadata; it does not create signer identity or consent evidence. Use real, consented, pseudonymous IDs only. Never infer signer identity from folder names, filenames, appearance, or model outputs.

## Fields

| Field | Type | Contract |
|---|---|---|
| signer_id | string | Pseudonymous participant ID, e.g. S001. Repeats across that signer's samples. |
| recording_id | string | Recording/session ID. Repeats for samples from one recording and must map to one signer. |
| sample_id | string | Unique ID for each row across the manifest. |
| label | string | One uppercase letter A-Z. |
| source_path | string | Relative path to the source image under the input root. |
| image_sha256 | string | 64-character hexadecimal SHA-256 of the source image bytes. |
| split | string | unassigned for a manifest awaiting grouped splitting, or an explicit train, val, or test assignment. |
| capture_metadata | object | Includes all seven keys listed below. Unknown values must be null, not guessed. |
| consent_id | string | Non-empty reference to the consent record. Do not store private consent documents in the manifest. |
| landmark_status | string | One of ok, no_hand, low_conf, failed. Preparation updates it from extraction results. |
| failure_reason | string or null | Required and non-empty when status is no_hand, low_conf, or failed; otherwise normally null. |

Required capture_metadata keys:

- device
- lighting
- background
- distance_cm_bucket
- hand_dominance
- resolution
- fps

Each capture value can be null when it was not collected. fps, when known, must be a positive number.

## Split and identity rules

- sample_id is globally unique. Repeated signer and recording IDs are expected; a recording may not map to multiple signers.
- A manifest must be wholly unassigned or wholly assigned. Mixed states are rejected.
- For explicit assignments, every signer must occur in one split only.
- With all rows set to split = unassigned, the preparation script assigns signer groups to train/validation/test using the recorded seed. The default target proportions are 70% / 15% / 15%; label coverage is optimized where the observed signer-label distribution permits it.
- The active RealSign archive currently has no verified signer IDs. It cannot be converted into valid signer metadata by this tooling. Signer-independent evaluation remains blocked until genuine consented signer-aware data exists.

## Validation

From the repository root:

    PYTHONPATH=backend python backend/scripts/validate_manifest.py --manifest /path/to/manifest_v2.jsonl

Example shape (illustrative IDs only; not a real dataset row):

    {"signer_id":"S001","recording_id":"R001","sample_id":"R001-0001","label":"A","source_path":"Training (A-Z)/A/frame-0001.jpg","image_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","split":"unassigned","capture_metadata":{"device":null,"lighting":null,"background":null,"distance_cm_bucket":null,"hand_dominance":null,"resolution":null,"fps":null},"consent_id":"CONSENT-REFERENCE-001","landmark_status":"failed","failure_reason":"not_processed"}

The example identifiers are placeholders. Never copy them as real provenance.
