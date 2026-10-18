# PulseSplit — Deep Learning Heart–Lung Sound Separation

> **Research prototype only.** Separates single-channel cardiopulmonary recordings into estimated heart and lung signals. Not a diagnostic device.

## Dataset policy

**Only approved source:** [UCI HLS-CMDS, dataset 1202](https://archive.ics.uci.edu/dataset/1202/hls-cmds). No data from other repositories is used for model training.

Torabi, Y., Shirani, S., & Reilly, J. (2025), *Descriptor: Heart and Lung Sounds Dataset Recorded From a Clinical Manikin Using Digital Stethoscope*, IEEE Data Descriptions, DOI: [10.1109/IEEEDATA.2025.3566012](https://doi.org/10.1109/IEEEDATA.2025.3566012). UCI dataset license: **CC BY 4.0**; attribute the authors and UCI. Raw files are intentionally excluded from Git and the ZIP.

The archive lists 535 WAV recordings (50 heart-only, 50 lung-only, and 145 mixed triplets of mixed/heart/lung audio). All recordings use 22,050 Hz and 15 seconds; we resample to 8,000 Hz for efficient training. We **train on the 145 matched mixed-source triplets**, not on all 535 recordings as independent examples.

## Method

- UCI WAV archive -> paired `M####.wav`, `H####.wav`, `L####.wav` discovery
- **70/15/15 recording-level split** (seed 42); no segment overlaps across train/val/test
- 8 kHz mono, 2-second segments, shared per-triplet scaling
- STFT (512-point FFT / 128-hop), compact 2-D U-Net, two softmax masks, log-magnitude L1 objective
- Best validation checkpoint saved; final evaluation runs **once** on held-out recording IDs
- SI-SDR (heart/lung) compared with using raw mixture as both predictions; per-recording CSV
- Streamlit demo to upload a mixed WAV and play/visualize estimates

**Important:** the source signals are separately recorded, and waveform phase/timing may not perfectly coincide with the mixture. Magnitude-domain training mitigates but does not remove this issue; report both subjective listening and rigorous held-out metrics. Clinical/general-purpose validation is outside project scope.

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate  |  Linux/macOS: source .venv/bin/activate
pip install -e ".[test,demo]"
pulsesplit download                  # downloads ONLY from official UCI
pulsesplit inspect
pulsesplit train --epochs 30 --batch-size 8
pulsesplit evaluate
streamlit run app.py
pytest -q
```

If UCI download is blocked by the environment, download `HLS-CMDS.zip` yourself from the linked UCI page and run:

```bash
pulsesplit download --archive /path/to/HLS-CMDS.zip
```

## Files

- `src/pulsesplit/audio.py`: WAV loading, resampling, source metrics
- `src/pulsesplit/data.py`: recursive nested ZIP extraction, pairing, leakage-aware splits
- `src/pulsesplit/model.py`: STFT and magnitude-mask U-Net
- `src/pulsesplit/pipeline.py`: training, validation, test evaluation, persisted metrics
- `src/pulsesplit/cli.py`: dataset/download/inspection/train/evaluate commands
- `app.py`: 3-channel playback / spectrogram Streamlit UI
- `tests/`: synthetic **unit-test fixture only** (not real model training results)
- `docs/`: report and experiment protocol

## Report status

The editable report is a **research/project draft** with the reference PDF's academic format. Results tables are deliberately marked **PENDING REAL UCI RUN** until the official dataset can be downloaded and training/evaluation has been executed. Do not represent unit-test fixture outputs as scientific results.
