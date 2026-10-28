# Temporal decision

**Decision: not testable with current data. STOP temporal model work for this dataset revision.**

The repository's manifest schema v2 includes recording_id, but no frame_index or timestamp/order field. The existing dataset preparation path calls MediaPipe in image mode and records independent labelled image rows. The active RealSign source has no verified signer IDs and the raw dataset is not checked into this repository, so this repository audit cannot assert that hidden external data contains sequences.

No temporal head, smoothing policy, GRU, or 1D convolution has been added. A temporal experiment must not be synthesized from alphabet-labelled still images or filenames.

## Diagnostic

Run the diagnostic against private/local manifests when available:

    PYTHONPATH=backend python backend/scripts/diagnose_temporal_sequences.py --manifest-root /path/to/dataset-manifests --output-json /path/outside/repo/temporal_diagnostic.json

The script requires labelled rows with both recording_id and numeric frame_index. It groups by recording and label, rejects duplicated frame indexes within a group, and only reports candidate sequences when at least two distinct ordered frames exist. Synthetic unit fixtures verify both the detected-sequence and no-sequence paths; these fixtures are not evidence that production data has sequences.

Required before revisiting temporal modelling:
- real frame sequences with stable A-Z labels and explicit frame indexes;
- recording and signer IDs grounded in consented metadata, never inferred from filenames or appearance;
- signer-level train/validation/test separation;
- validation comparison against the same per-frame baseline across at least three seeds;
- test split opened once only after validation selection;
- measured inference latency within an explicitly set budget.

Until these conditions are met, temporal validation results are **NOT RUN**, not zero and not inferred from results for another sign language.
