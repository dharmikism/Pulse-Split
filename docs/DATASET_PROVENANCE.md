# Provenance and dataset restrictions

## Allowed source: UCI Machine Learning Repository
- Dataset: HLS-CMDS, UCI ID 1202.
- Website: https://archive.ics.uci.edu/dataset/1202/hls-cmds
- Dataset creators: Yasaman Torabi, Shahram Shirani, James P. Reilly.
- Publication: IEEE Data Descriptions (2025), DOI: 10.1109/IEEEDATA.2025.3566012.
- Reported license: CC BY 4.0. Attribution is required.
- 535 WAV files: 50 isolated heart, 50 isolated lung, 145 mixed+paired heart+paired lung triplets.
- Recordings: 15s, 22050 Hz.

No data from Kaggle, GitHub, Zenodo or other repositories are included in the training pipeline. Github and research papers can be consulted for documentation only. Simulated sine waves in `tests/` are strictly synthetic unit-test inputs and MUST NOT be described as experimental data.

## Outstanding validations
- Confirm exact raw file paths and naming after original UCI archive download.
- Verify samples, duration, channels, metadata columns and audio waveform quality.
- Check synchronization/phase mismatches between individually recorded components and mixed audio.
- Decide whether stricter sound-type or sequence grouping is required to mitigate repeated simulator playback leakage.
- Record SHA256 hash of the downloaded UCI archive for exact reproducibility.
