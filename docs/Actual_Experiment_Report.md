# PulseSplit: recorded dataset experiment

**Evaluated data label (user supplied): UCI HLS-CMDS**

Evaluation completed (UTC): 2026-10-19T11:41:28.505738+00:00

Held-out test recordings: **22**. Checkpoint and split hashes, test IDs, and aggregate statistics were cross-checked against saved artifacts.

## Held-out SI-SDR results

All scores and gains are in dB. Gain is separated-estimate SI-SDR minus the unprocessed-mixture baseline for the same recording and source. Positive gain means an improvement; negative gain means the estimate scores below the mixture. SI-SDR is not percentage accuracy.

The 95% intervals below are descriptive per-recording bootstrap intervals. Shared-source dependence is not modeled; these are not calibrated intervals over independent source components.

| Source / measurement | Valid n | Undefined | Mean (dB) | Median (dB) | SD (dB) | Mean 95% CI (dB) |
| --- | --- | --- | --- | --- | --- | --- |
| Heart: mixture baseline | 22 | 0 | -28.201 | -33.632 | 20.051 | [-36.245, -19.863] |
| Heart: separated estimate | 22 | 0 | -25.010 | -32.269 | 15.768 | [-31.370, -18.417] |
| Heart: gain over mixture | 22 | 0 | 3.191 | 2.216 | 7.771 | [-0.034, 6.504] |
| Lung: mixture baseline | 22 | 0 | -27.069 | -34.814 | 23.157 | [-36.476, -17.592] |
| Lung: separated estimate | 22 | 0 | -26.909 | -36.335 | 21.779 | [-35.558, -17.625] |
| Lung: gain over mixture | 22 | 0 | 0.160 | -0.453 | 4.791 | [-1.268, 2.424] |

Statistics are calculated over complete held-out recordings, not training windows. The intervals use a percentile bootstrap of recording means (2,000 resamples; seed 42). Undefined scores are excluded and counted; intervals are unavailable with fewer than two valid recordings. Rounded report values are backed by full-precision JSON and CSV outputs.

These intervals assume independent recordings. Repeated manikin playback patterns may violate that assumption; the interval does not capture retraining variability or clinical generalization.

## Training and experiment configuration

| Setting | Recorded value |
| --- | --- |
| Training data label (user supplied) | UCI HLS-CMDS |
| Recordings | 145 |
| Train / validation / test recording counts | 101 / 22 / 22 |
| Split strategy | exact-source-connected-components |
| Completed epochs | 10 |
| Selected checkpoint epoch | 3 |
| Selected checkpoint training log-magnitude MAE | 0.09426313062241931 |
| Selected checkpoint validation log-magnitude MAE | 0.07992566674463934 |
| Exact-source connected components | 28 |
| Train / validation / test component counts | 14 / 5 / 9 |
| Distinct byte hashes (mixture / heart / lung) | 144 / 89 / 74 |
| Batch size | 8 |
| U-Net base channels | 16 |
| Training segment length (seconds) | 2.0 |
| Model parameters | 488434 |
| Seed | 42 |
| Training device | cpu |
| Training runtime (seconds) | 428.0974077999708 |
| Training completed (UTC) | 2026-10-19T11:41:24.809017+00:00 |
| Python | 3.14.2 |
| Platform | Windows-11-10.0.26200-SP0 |
| PyTorch | 2.14.1+cpu |
| NumPy | 2.5.4 |

Checkpoint selection uses validation loss with a minimum improvement of 0.00001. The selected checkpoint epoch can precede the final completed epoch. The training objective is log-magnitude spectral mean absolute error; it is different from the SI-SDR evaluation metric.

## Completed training history

| Epoch | Training log-magnitude MAE | Validation log-magnitude MAE |
| --- | --- | --- |
| 1 | 0.09483767002610202 | 0.08117930325014251 |
| 2 | 0.09456351774585298 | 0.08007285206929429 |
| 3 | 0.09426313062241931 | 0.07992566674463934 |
| 4 | 0.09382357700204917 | 0.08007866790058552 |
| 5 | 0.09423604020598586 | 0.08010517873547295 |
| 6 | 0.09403779121613738 | 0.07992941228213249 |
| 7 | 0.09380025537803116 | 0.08105962568676317 |
| 8 | 0.09367490125638255 | 0.08024208412154929 |
| 9 | 0.09333306552033445 | 0.08050352804266013 |
| 10 | 0.09351550610643802 | 0.08016256750984625 |

## Native audio audit

| Native audio measurement | Audited value |
| --- | --- |
| Matched WAV files | 435 |
| Native sample rate(s), Hz | 4,000 |
| Channel count(s) | 1 |
| File duration range, seconds | 15.0 to 15.0 |
| Total mixture duration, seconds | 2175.0 |
| Audit status | warnings |

The measured native rate(s), 4,000 Hz, differ from the 22,050 Hz stated in the [official UCI description](https://archive.ics.uci.edu/dataset/1202/hls-cmds). Report the measured file properties for this downloaded release rather than substituting the catalogue rate.

The model pipeline resamples to 8,000 Hz. Upsampling lower-rate source audio does not restore frequencies absent from the original recording.

Audit issue counts: possible_clipping = 36. The possible_clipping flag marks near-full-scale samples under the recorded threshold; it is a screening warning, not proof of signal distortion.

## Dataset provenance and interpretation limits

UCI describes HLS-CMDS as digital-stethoscope recordings from a clinical manikin, including mixed recordings and corresponding source recordings. These are manikin recordings, not a patient cohort. [Official UCI dataset description](https://archive.ics.uci.edu/dataset/1202/hls-cmds).

The data labels in run_metadata.json and summary.json are supplied by the experiment runner. The report verifies consistency of the saved experiment files; it does not independently authenticate the downloaded archive or assign a clinical meaning to the scores. Retain the download URL, archive SHA-256, audit outputs, and original dataset metadata with the submission.

Corresponding IDs do not by themselves prove sample-level synchronization or that the mixture is exactly the sum of the two reference recordings. When source components are recorded in separate acquisition phases, time/phase differences can strongly affect waveform SI-SDR. The current pipeline resamples audio, trims to a shared length, and uses shared amplitude scaling; it does not perform lag or phase correction. Treat scores under this protocol as an initial benchmark, and inspect acquisition/alignment evidence before drawing source-fidelity conclusions.

Partitions are assigned before segmentation. Recorded split strategy: exact-source-connected-components. Recording IDs connected through byte-identical mixed, heart, or lung files are held in the same partition. This limits exact-source reuse across partitions. It is not a holdout of all simulator sound types or acoustic patterns. Shared sources within a partition also mean recording-bootstrap intervals need not have calibrated coverage for independent sources. Repeated sound types, simulator playback patterns, and anatomical locations can still span partitions. A stricter grouping study and independent patient recordings are required for broader claims. No diagnosis, treatment, or clinical validation is established by this experiment. The pretraining protocol records a revision after duplicate-source inspection and before training/test evaluation; see results/PRETRAINING_PROTOCOL.md for the fixed procedure.

Dataset attribution: Yasaman Torabi, Shahram Shirani, and James P. Reilly (2025), HLS-CMDS; CC BY 4.0. [UCI attribution and license](https://archive.ics.uci.edu/dataset/1202/hls-cmds).

## Recorded archive provenance

Dataset source: https://archive.ics.uci.edu/static/public/1202/hls-cmds%3A%2Bheart%2Band%2Blung%2Bsounds%2Bdataset%2Brecorded%2Bfrom%2Ba%2Bclinical%2Bmanikin%2Busing%2Bdigital%2Bstethoscope.zip

Original archive SHA-256: `183ddb847c70dc394120284c2c3cb79c717fc37ac7f81bd29f3d5e5b572d4804`

This archive hash is preserved from the download workflow. Raw audio is not required to regenerate this report; the report does not re-download or re-authenticate it.

## Held-out recording IDs

`0007`, `0008`, `0011`, `0027`, `0035`, `0039`, `0048`, `0061`, `0064`, `0065`, `0072`, `0074`, `0089`, `0092`, `0095`, `0114`, `0117`, `0119`, `0129`, `0138`, `0139`, `0141`

Full per-recording results: results/test_per_recording.csv. Full summary and intervals: results/summary.json. Training curve: artifacts/training_history.png. Evaluation chart: results/test_comparison.png.

## Exact source artifacts

SHA-256 hashes identify the inputs used for this report. They do not certify experimental quality.

| Source artifact | SHA-256 |
| --- | --- |
| artifacts/best.pt | `8f7d663dcf0b582291c535c2e2935afb5c2f348ed67c18e6fe78f9f5f9f8bce9` |
| artifacts/run_metadata.json | `b8741ede424311c6a6b755f9eba0e207ed3476136d370c0d650c35b4f0f3055c` |
| artifacts/history.json | `bc7fb5c545f35c83c88018603e110fb3803a98191b8e8b91884b5a0c97a2e7b1` |
| artifacts/split.json | `a03c65a388c4033311a4fcfda76b95b4ac3c091cb17953ca02a3980d80bf7bb1` |
| results/summary.json | `fffcf839198c9bd7d1d88f7a6af5dcb53d1ff87bc01feaa90885900f5f7d9170` |
| results/test_per_recording.csv | `a7c673536c2376f2573e6126179a524e848bd203a6c793fcada2501520921269` |
| results/dataset_provenance.json | `9e40d3e2851af1317cfa9754ba4ba12f772bd0913b54265d0f12c1519f3e96f9` |
| results/audio_sha256.json | `24db74754b7c46081d652e2785b0c764296a3c9d07d9130cb75180be27b96629` |
| results/audit/dataset_audit.json | `35702c4a742bd8d513f96645314d3fac776c47a19ee8d1dacf2d2d40ef6f3dad` |
| results/PRETRAINING_PROTOCOL.md | `fd08548e51fdbc6f1eef21f6eae1ea5fbc0a7a510a61ee2e9e7c51b081bad03b` |

## Regenerate this appendix

```powershell
python docs/build_experiment_report.py --artifacts artifacts --results results --output results/actual_experiment_report
```

The Word appendix remains editable. Existing working-draft reports are not overwritten.
