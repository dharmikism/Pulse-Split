# Stored demonstration recordings

Source: Torabi, Y., Shirani, S., & Reilly, J. (2025), *Heart and Lung Sounds
Dataset Recorded From a Clinical Manikin Using Digital Stethoscope*.
UCI HLS-CMDS, https://archive.ics.uci.edu/dataset/1202/hls-cmds
DOI: https://doi.org/10.1109/IEEEDATA.2025.3566012
License: Creative Commons Attribution 4.0 International,
https://creativecommons.org/licenses/by/4.0/

The mixture files are mono recordings resampled to 8 kHz and normalized.
Heart and lung files are model-generated estimates derived from those mixtures,
not ground-truth source recordings. All files are float WAVs. The manifest records
the checkpoint SHA-256 used to generate them. The three examples are the first,
middle, and last sorted held-out test IDs, selected without reference to scores.

These clinical manikin examples demonstrate the software. They are not clinical
validation or evidence of performance on patient recordings.
Regenerate using `python tools/prepare_demo_examples.py` from the project root.
