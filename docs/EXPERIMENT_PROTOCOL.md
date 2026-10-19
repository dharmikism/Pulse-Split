# Experiment plan — preregistered before seeing test performance

1. Verify original UCI archive and enumerate `M0001` to `M0145` with matching `H` and `L` WAV files.
2. Inspect sampling rates, lengths, silence, clipping and 1-to-1 pairing.
3. Split recording IDs 70/15/15 (fixed seed 42) BEFORE segmenting.
4. Validate frequency and temporal alignment and document if audio tracks require alignment beyond simple trimming. No tuning using test sources.
5. Train a compact magnitude-mask U-Net (30 epochs max, early stopping patience 7) and a frequency-mask baseline based on low-/high-frequency passbands, if empirically justified by training-only analysis.
6. Hold test untouched until model and hyperparameters fixed; report heart/lung SI-SDR and improvement over direct mixture baseline, with per-recording distributions and examples.
7. Run ablations: (a) smaller U-Net base channels; (b) alternate STFT hop; (c) shared vs no energy normalization; declare these exploratory.
8. Record package versions, hardware and runtime. Export figures directly from results, never redraw reported values manually.
9. Describe external validity limitation: data recorded on simulator under controlled conditions; no clinical deployment.

## Leakage warning
Audio segments from the same 15-second original recording MUST NOT be in separate splits. If different recordings are generated from the same original simulator playback pattern, even recording-level splits may still share repeated underlying patterns; evaluate cross-pattern holdouts if metadata allows.

## Actual archive pretraining amendment (19 October 2026)

The official downloaded archive contains 145 matched triplets but only 144 unique
mixture files, 89 unique heart files, and 74 unique lung files by SHA-256. The
original recording-ID split can therefore share identical audio across partitions.
Before training or evaluating the real model, replace it with connected-component
grouping: link any recording IDs that share a byte-identical mixture, heart or lung
file, and assign whole components to train/validation/test with seed 42. Save the
actual split counts, component mapping and file hashes in `artifacts/split.json`.
This addresses exact duplicates; related simulator patterns may remain shared.

Matched WAV headers show mono, 4,000 Hz, and 15 seconds, despite the catalogue's
22,050 Hz description. Keep 8,000 Hz internal preprocessing as planned; this
upsampling does not add recorded bandwidth. Preserve all native-data audit warnings.

The provided source tracks are separately recorded and are not verified as
sample-aligned additive stems. Report raw reference SI-SDR with this limitation;
do not interpret it as clinical accuracy. Freeze the model configuration at base
16, two-second segments, batch size 8, seed 42, up to 30 epochs and validation
patience 7 before training. Select using validation loss and evaluate the final
test split once. No test-score-driven tuning is part of this run. Ablations and
additional baselines in the earlier plan remain future work.
