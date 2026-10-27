# PulseSplit — Deep Learning Heart–Lung Sound Separation

> **Research prototype only.** Separates single-channel cardiopulmonary recordings into estimated heart and lung signals. Not a diagnostic device.

## Dataset policy

**Only approved source:** [UCI HLS-CMDS, dataset 1202](https://archive.ics.uci.edu/dataset/1202/hls-cmds). No data from other repositories is used for model training.

Torabi, Y., Shirani, S., & Reilly, J. (2025), *Descriptor: Heart and Lung Sounds Dataset Recorded From a Clinical Manikin Using Digital Stethoscope*, IEEE Data Descriptions, DOI: [10.1109/IEEEDATA.2025.3566012](https://doi.org/10.1109/IEEEDATA.2025.3566012). UCI dataset license: **CC BY 4.0**; attribute the authors and UCI. Dataset WAVs stay outside Git. The teacher submission ZIP includes the 435 matched WAVs with attribution and license information.

The archive lists 535 WAV recordings (50 heart-only, 50 lung-only, and 145 mixed triplets of mixed/heart/lung audio). The catalogue states 22,050 Hz, but the downloaded matched WAV headers show **4,000 Hz, mono, 15 seconds**. The pipeline resamples to 8,000 Hz; upsampling does not restore missing bandwidth. We **train on the 145 matched mixed-source triplets**, not on all 535 recordings as independent examples.

## Method

- UCI WAV archive -> paired `M####.wav`, `H####.wav`, `L####.wav` discovery
- **Source-group split targeting 70/15/15** (seed 42): keep connected recordings with byte-identical mixture/heart/lung WAVs together before segmentation
- 8 kHz mono, 2-second segments, shared per-triplet scaling
- STFT (512-point FFT / 128-hop), compact 2-D U-Net, two softmax masks, log-magnitude L1 objective
- Best validation checkpoint saved; final evaluation runs **once** on held-out recording IDs
- SI-SDR (heart/lung) compared with using raw mixture as both predictions; per-recording CSV
- Streamlit demo to upload a mixed WAV and play/visualize estimates
- Compare waveform or spectrogram views, inspect recording details, and download
  individual float WAV estimates or a ZIP containing all audio and metadata
- Validate uploads for duration, size, silence, channels, and non-finite samples

**Important:** the source signals are separately recorded, and waveform phase/timing may not perfectly coincide with the mixture. Magnitude-domain training mitigates but does not remove this issue; report both subjective listening and rigorous held-out metrics. Clinical/general-purpose validation is outside project scope.

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate  |  Linux/macOS: source .venv/bin/activate
pip install -e ".[test,demo,reports]"
pulsesplit download                  # downloads ONLY from official UCI
pulsesplit inspect
pulsesplit audit                    # JSON/CSV audio quality checks
pulsesplit train --epochs 30 --batch-size 8 --dataset-label "UCI HLS-CMDS"
pulsesplit evaluate
streamlit run app.py
pytest -q
```

To execute the complete real-data protocol with saved provenance, audit, training
log, one final evaluation, an editable report, and a results ZIP:

```bash
python tools/run_real_experiment.py
```

Run this after `pulsesplit download`, with empty experiment output directories.
The official archive has reused source files, so random recording-ID splits can
share identical audio. The real run uses 28 connected components and a 101/22/22
recording split. Related simulator patterns may still span partitions. The raw
reference recordings are not verified sample-aligned additive stems, so interpret
waveform SI-SDR cautiously. See `docs/EXPERIMENT_PROTOCOL.md` for the pretraining amendment.

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

## Build a submission with execution evidence

For the teacher layout with `dataset/`, `output/results/`, `output/models/`,
`output/charts/`, and an executed notebook, see [what to submit](docs/SUBMISSION_GUIDE.md).
Run `python tools/package_teacher_submission.py` after the actual experiment,
with `.[notebook]` dependencies installed. This submission includes the 435
matched dataset WAVs with UCI attribution and license. Generated audio, models,
and ZIPs remain outside Git.

After installing `.[test,demo,reports]`, run:

```bash
python tools/build_submission.py
```

This runs the unit tests, trains a small model for two epochs on nine synthetic
audio fixtures, reloads its checkpoint, checks inference, and creates
`submission/<UTC timestamp>/PulseSplit_Submission.zip`. Send that ZIP to your teacher.
It includes source code, tests, reports, architecture figures, and a `proof/`
folder with test logs, training history, environment information, and a clearly
labeled plot. An editable `proof/Software_Verification_Report.docx` reads actual
test counts and synthetic loss values from the same run. `SHA256SUMS.json` records
the packaged file hashes. The synthetic audit and evaluation exports are included
as software checks and are labeled accordingly.

The verification model is temporary. Synthetic checks demonstrate software execution;
they do **not** demonstrate real UCI separation quality. For real experiment proof,
follow the download/train/evaluate commands above and additionally share
`artifacts/history.json`, `artifacts/run_metadata.json`, `artifacts/split.json`,
and the generated files in `results/`. The synthetic verification ZIP excludes raw
data; the teacher-layout ZIP explicitly includes the matched dataset. Keep `.venv/`,
`.git/`, caches and credentials out of both submissions.

## Experiment evidence and report exports

`pulsesplit audit --splits artifacts/split.json` checks native sample rates,
duration differences, silent channels, possible clipping, corrupt/nonfinite audio,
and recording overlap. Read `results/audit/dataset_audit.json` and the per-file CSV
before interpreting model results. The audit checks files; it does not verify their origin.

Training now saves `artifacts/training_history.png` plus runtime, parameter count,
configuration, environment and data label in `run_metadata.json`.
Use `--dataset-label` to identify the supplied data; labels are user supplied.

Evaluation rejects overlapping/invalid split IDs and exports `results/evaluation_report.md`.
It reports per-recording means, medians, standard deviations, valid/undefined score
counts, and deterministic bootstrap 95% confidence intervals for mean scores.
Intervals assume independent recordings and do not address repeated simulator
patterns or clinical generalization. Undefined scores are exported as JSON `null`.
Checkpoint and split hashes identify the evaluated artifacts.

See [software verification reports](docs/SOFTWARE_VERIFICATION.md) to regenerate
an editable report from a particular evidence directory.

## Research results status

The first actual UCI run completed on 19 October 2026: **101 training / 22 validation /
22 test recordings**, grouped to prevent byte-identical audio crossing partitions.
Training stopped after 10 epochs, selecting epoch 3; the model has 488,434 parameters.

| Source | Mixture SI-SDR (dB) | Estimate SI-SDR (dB) | Mean gain (dB) |
|---|---:|---:|---:|
| Heart | -28.201 | -25.010 | +3.191 |
| Lung | -27.069 | -26.909 | +0.160 |

These low raw-reference SI-SDR scores are affected by unverified reference
time/phase alignment. Both descriptive recording-bootstrap gain intervals include
zero; they also do not account for within-component dependence. Do not present
these values as percentage accuracy or evidence of clinical usefulness.

See the measured [actual experiment report](docs/Actual_Experiment_Report.md) and
its editable [Word version](docs/Actual_Experiment_Report.docx). The original
working-draft Word/PDF retains its earlier placeholders; use the actual report for
measured results. Synthetic software verification remains separate.

## Frontend with stored examples

Launch the local audio lab with `streamlit run app.py`. The default **Saved examples**
view includes three held-out UCI manikin recordings and precomputed model estimates;
playback, waveform/spectrogram views, and WAV/ZIP downloads work without loading a
checkpoint. Switch to **Upload WAV** to separate your own audio using `artifacts/best.pt`.

To regenerate examples after training, run `python tools/prepare_demo_examples.py`.
This selects the first, middle, and last sorted test IDs, rather than selecting by
performance. The manifest records the checkpoint hash. Example audio is distributed
under the source dataset's CC BY 4.0 license; see [examples/README.md](examples/README.md).
