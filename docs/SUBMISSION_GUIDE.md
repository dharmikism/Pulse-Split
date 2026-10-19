# What to submit

Send the generated `PulseSplit_Submission.zip`. It contains two top-level folders,
the executed notebook, and a short README:

```text
PulseSplit_Submission/
├── dataset/
│   ├── mixed/                  # 145 mixture WAVs
│   ├── heart/                  # 145 corresponding heart WAVs
│   ├── lung/                   # 145 corresponding lung WAVs
│   ├── metadata/               # Original UCI metadata CSVs
│   ├── recordings.csv          # File index and train/val/test partition
│   ├── provenance.json
│   └── README.md               # Source, attribution and CC BY 4.0 license
├── output/
│   ├── results/                # Report, metrics, audit, logs and verification
│   ├── models/                 # best.pt, split, configuration and Python source
│   └── charts/                 # Training loss and held-out comparison
├── PulseSplit.ipynb            # Executed notebook with actual outputs
└── README.md
```

The dataset folder contains the **435 matched WAVs actually used** in this run,
from the full 535-file official release. The other 100 isolated recordings were
not used. Metadata, authorship and dataset license accompany the audio.

Open the notebook first. Read `output/results/Actual_Experiment_Report.docx` for
the editable report and `output/results/test_per_recording.csv` for every held-out
score. The notebook reads the saved actual test results and demonstrates one
validation recording; it does not retrain or recompute test scores.

The model ran for 10 epochs and selected epoch 3. Raw-reference SI-SDR is low;
reference timing/phase and simulator-pattern dependence limit its interpretation.
Neither the scores nor the software-test counts are percentage accuracy.

To rebuild after completing the real experiment:

```powershell
python -m pip install -e ".[test,demo,reports,notebook]"
python tools/package_teacher_submission.py
```

The tool prints the ZIP path only after verifying model/result consistency, copied
audio hashes, notebook execution and ZIP integrity. It installs a notebook kernel
inside the active Python environment. The generated dataset and trained model stay
inside the ignored `submission/` directory; the source code, report and notebook
can be pushed to GitHub separately.
