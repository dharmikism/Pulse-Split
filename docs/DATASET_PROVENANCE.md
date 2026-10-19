# Provenance and dataset restrictions

## Allowed source: UCI Machine Learning Repository
- Dataset: HLS-CMDS, UCI ID 1202.
- Website: https://archive.ics.uci.edu/dataset/1202/hls-cmds
- Dataset creators: Yasaman Torabi, Shahram Shirani, James P. Reilly.
- Publication: IEEE Data Descriptions (2025), DOI: 10.1109/IEEEDATA.2025.3566012.
- Reported license: CC BY 4.0. Attribution is required.
- 535 WAV files: 50 isolated heart, 50 isolated lung, 145 mixed+paired heart+paired lung triplets.
- Catalogue description: 15s, 22050 Hz. **Actual matched WAV headers in the downloaded release: 15s, mono, 4000 Hz.** Internal preprocessing resamples to 8000 Hz and adds no recorded bandwidth.

No data from Kaggle, GitHub, Zenodo or other repositories are included in the training pipeline. Github and research papers can be consulted for documentation only. Simulated sine waves in `tests/` are strictly synthetic unit-test inputs and MUST NOT be described as experimental data.

## Outstanding validations
- Confirm exact raw file paths and naming after original UCI archive download.
- Verify samples, duration, channels, metadata columns and audio waveform quality.
- Check synchronization/phase mismatches between individually recorded components and mixed audio.
- Decide whether stricter sound-type or sequence grouping is required to mitigate repeated simulator playback leakage.
- Record SHA256 hash of the downloaded UCI archive for exact reproducibility.

## Actual archive inspection (19 October 2026)

The download contains all 145 matched triplets. SHA-256 comparison finds 144
unique mixture files, 89 unique heart files and 74 unique lung files. Comparing
hashes across all roles connects these into 28 components. The real experiment
keeps entire components together, using 101 train / 22 validation / 22 test
recordings (14 / 5 / 9 components, respectively). Exact duplicate M0111/M0126
stays within validation. Hash grouping prevents byte-identical audio crossing
partitions, not all shared simulator templates.

The native audio audit flags 36 matched files for possible near-full-scale
clipping. This is a screening warning rather than proof of waveform distortion.
Separately recorded reference signals are not verified synchronized additive
stems; waveform SI-SDR must be interpreted with that limitation. The exact archive
and per-file hashes are saved under `results/` during the actual experiment.
