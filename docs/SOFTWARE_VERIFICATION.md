# Editable evidence report

`build_verification_report.py` turns one completed submission run's evidence into
an editable Word appendix and a Markdown copy. Every reported count and loss
comes from that run's artifacts. It never copies numbers from an earlier run.

Install the optional report dependency from the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[reports]"
```

Build a fresh submission with execution evidence:

```powershell
.\.venv\Scripts\python.exe tools/build_submission.py
```

To generate the report from an existing completed run, replace `<run>` with its
directory name:

```powershell
.\.venv\Scripts\python.exe docs/build_verification_report.py "submission/<run>/proof"
```

The output files are `Software_Verification_Report.docx` and
`Software_Verification_Report.md` in that proof directory. Open the DOCX in Word,
Google Docs, or LibreOffice to edit it or export a PDF. `--markdown-only` works
without the optional Word dependency.

The generator requires `tests.xml`, `verification.json`, `history.json`,
`run_metadata.json`, and `split.json`. It checks split overlap, recording totals,
finite consecutive epoch losses, the selected checkpoint's validation loss, and agreement between
JUnit failures and the verifier's pass claim. Source hashes identify the exact
artifacts used for the report.

## Numbers you may use

Use the generated table for **software verification on synthetic audio**:
the actual passed/failed/skipped test counts, completed epoch count, synthetic
recording split, exact recorded training/validation losses, checkpoint reload
status, and inference dimensions. A smaller synthetic loss only describes the
generated fixture experiment. It is not a percentage accuracy or real-world
separation score.

Keep the main experimental table pending until you run the documented real-data
training and held-out evaluation. Report heart and lung SI-SDR and improvement
over the mixture baseline with dataset provenance and reproducible settings.

`PulseSplit_Report_Working_Draft.docx` and its PDF are background working drafts.
This generator produces a separate evidence appendix; it does not update those
drafts or imply that their PDF has been regenerated.
