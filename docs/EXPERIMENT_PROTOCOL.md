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
