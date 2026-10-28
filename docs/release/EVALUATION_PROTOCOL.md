# Held-out signer evaluation protocol

## Status

Signer-independent evaluation is BLOCKED for the current RealSign archive because verified signer IDs are not available. Do not infer signer IDs or describe a source-folder split as a signer split. This protocol becomes executable only after new, consented data with real pseudonymous signer and recording metadata has been collected.

## Partitioning protocol

1. Validate manifest v2. Reject missing or duplicate sample IDs, missing consent references, invalid labels, and any signer appearing across multiple assigned partitions.
2. If every row is split = unassigned, assign whole signers with the preparation script's deterministic seed. Record the seed and target proportions (default 70% train, 15% validation, 15% test) in labels.json. Stratify by A-Z label coverage where supported by the signer-level data. Do not move individual samples across a signer's partition.
3. Use train signers for fitting. Use validation signers for architecture/hyperparameter selection, early stopping, and threshold choices. Do not inspect test metrics, examples, confusion patterns, or errors during these decisions.
4. Freeze the candidate checkpoint and its SHA-256 before test evaluation. Evaluate the untouched test signers once per candidate promotion decision. If a candidate is rejected, do not tune against that test result or repeatedly re-score candidates on the same held-out test signers. Collect a fresh independent test cohort for a future promotion decision.
5. Report classification accuracy alongside landmark extraction failure rate. Classification accuracy is conditional on samples with usable landmarks; the extraction-failure rate measures the separate front-end data loss and must not be hidden inside the classifier metric.
6. A cross-split exact or near-duplicate is a pipeline failure. Fix the data partition/leakage problem before evaluating a candidate. Never silently remove a test sample after looking at model predictions.

## Provenance record

Create one record for each promotion evaluation. Fill it using actual execution details; do not leave placeholders in a published report.

| Field | Value |
|---|---|
| Candidate/model version | |
| Checkpoint SHA-256 | |
| Manifest SHA-256 | |
| Dataset source/version | |
| Test signer-set or cohort identifier (pseudonymous) | |
| Evaluator (person or CI identity) | |
| Evaluation timestamp (UTC) | |
| Evaluator command and code commit SHA | |
| Test evaluation executed for this promotion before? | |
| Overall and macro accuracy | |
| Per-class accuracy and confusion matrix artifact | |
| Landmark extraction failure report artifact | |
| Duplicate-leakage check | |

Never fill a person, timestamp, checkpoint hash, manifest hash, or result from inference. Record the actual values at execution time. Protect participant privacy; the provenance record should use approved pseudonyms/cohort references, not names or direct personal identifiers.

## Current release status

The existing recorded 98.2733% overall result is a source-image-split metric from the stored report, not a signer-independent score. The active archive lacks verified signer IDs, so the signer-independent release gate remains BLOCKED until genuine new data and a fresh held-out signer cohort exist.
