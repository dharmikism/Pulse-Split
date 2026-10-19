"""Package the actual experiment as dataset/, output/, and an executed notebook."""
import csv
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from pulsesplit.data import find_triplets
from pulsesplit.reporting import file_sha256
from build_submission_notebook import build_notebook

ROOT = Path(__file__).resolve().parents[1]


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def main():
    import nbformat
    from nbclient import NotebookClient
    spec = importlib.util.spec_from_file_location('actual_report', ROOT / 'docs' / 'build_experiment_report.py')
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    # This checks consistency, hashes and saved metrics, without rerunning test inference.
    report.collect_experiment(ROOT / 'artifacts', ROOT / 'results')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    parent = ROOT / 'submission' / f'{stamp}_teacher'
    destination = parent / 'PulseSplit_Submission'
    dataset = destination / 'dataset'
    results = destination / 'output' / 'results'
    models = destination / 'output' / 'models'
    charts = destination / 'output' / 'charts'
    for folder in (dataset, results, models, charts):
        folder.mkdir(parents=True)
    splits = json.loads((ROOT / 'artifacts' / 'split.json').read_text())
    records = find_triplets(ROOT / 'data' / 'raw')
    if len(records) != 145:
        raise ValueError('Expected all 145 actual matched triplets')
    index = []
    for record in records:
        row = {'id': record.key}
        for role in ('mixed', 'heart', 'lung'):
            source = getattr(record, role)
            target = dataset / role / source.name
            copy(source, target)
            if file_sha256(target) != splits['file_hashes'][record.key][role]:
                raise ValueError(f'Audio differs from trained data: {source}')
            row[role] = target.relative_to(dataset).as_posix()
        row['partition'] = next(name for name in ('train', 'val', 'test') if record.key in splits[name])
        index.append(row)
    with (dataset / 'recordings.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['id', 'mixed', 'heart', 'lung', 'partition'])
        writer.writeheader()
        writer.writerows(index)
    for name in ('HS.csv', 'LS.csv', 'Mix.csv'):
        copy(ROOT / 'data' / 'raw' / name, dataset / 'metadata' / name)
    copy(ROOT / 'results' / 'dataset_provenance.json', dataset / 'provenance.json')
    (dataset / 'README.md').write_text(
        '# Official UCI HLS-CMDS audio used in the experiment\n\n'
        '435 WAV files = 145 mixed + 145 corresponding heart + 145 corresponding lung '
        'recordings. The 100 isolated recordings from the full 535-file dataset were not '
        'used in training and are not duplicated here. recordings.csv indexes the actual '
        'files and partitions. Native matched audio: mono, 4000 Hz, 15 seconds.\n\n'
        'Source: https://archive.ics.uci.edu/dataset/1202/hls-cmds\n\n'
        'Attribution: Yasaman Torabi, Shahram Shirani, James P. Reilly (2025). '
        'HLS-CMDS. UCI Machine Learning Repository. DOI: 10.1109/IEEEDATA.2025.3566012.\n\n'
        'Dataset license: Creative Commons Attribution 4.0 International (CC BY 4.0): '
        'https://creativecommons.org/licenses/by/4.0/\n\n'
        'These are clinical-manikin recordings, not patient data. Byte-identical audio '
        'is grouped before splitting. Reference timing/phase is not verified aligned.\n', encoding='utf-8')
    for name in ('best.pt', 'history.json', 'run_metadata.json', 'split.json'):
        copy(ROOT / 'artifacts' / name, models / name)
    for source in (ROOT / 'src' / 'pulsesplit').glob('*.py'):
        copy(source, models / 'source' / 'pulsesplit' / source.name)
    copy(ROOT / 'pyproject.toml', results / 'project_config.toml')
    copy(ROOT / 'LICENSE', models / 'source' / 'LICENSE')
    for name in ('summary.json', 'test_per_recording.csv', 'evaluation_report.md',
                 'dataset_provenance.json', 'audio_sha256.json', 'PRETRAINING_PROTOCOL.md',
                 'training_log.txt', 'requirements-freeze.txt', 'tests.xml', 'pytest.txt'):
        copy(ROOT / 'results' / name, results / name)
    for source in (ROOT / 'results' / 'audit').glob('*'):
        if source.is_file():
            copy(source, results / 'audit' / source.name)
    for suffix in ('.md', '.docx'):
        copy(ROOT / 'results' / 'report' / f'Actual_Experiment_Report{suffix}',
             results / f'Actual_Experiment_Report{suffix}')
    copy(ROOT / 'artifacts' / 'training_history.png', charts / 'training_history.png')
    copy(ROOT / 'results' / 'test_comparison.png', charts / 'test_comparison.png')
    (destination / 'README.md').write_text(
        '# PulseSplit submission\n\n'
        'Open PulseSplit.ipynb first: its cells have already been executed with the included '
        'actual-data files. For the editable report, open output/results/Actual_Experiment_Report.docx.\n\n'
        'dataset/: the exact 435 WAV files used, metadata, attribution, and recording index.\n'
        'output/results/: held-out metrics, report, audit, logs, test evidence, and file hashes.\n'
        'output/models/: trained checkpoint, training history, split, configuration, and Python source.\n'
        'output/charts/: training and test-comparison graphs.\n\n'
        'To rerun the notebook, install Python 3.10+ and: '
        '`python -m pip install numpy scipy soundfile torch pandas matplotlib scikit-learn ipykernel`. '
        'Select that Python kernel in Jupyter or VS Code. No network or retraining is needed '
        'after dependencies are installed.\n\n'
        'Results are SI-SDR in dB, not percentage accuracy. Absolute scores are low. '
        'Unverified reference alignment and repeated simulator patterns limit conclusions; '
        'both descriptive gain intervals include zero. Not for diagnosis.\n', encoding='utf-8')
    notebook_path = build_notebook(destination / 'PulseSplit.ipynb')
    subprocess.run([sys.executable, '-m', 'ipykernel', 'install', '--prefix', sys.prefix,
                    '--name', 'pulsesplit-submission', '--display-name', 'PulseSplit submission'], check=True)
    notebook = nbformat.read(notebook_path, as_version=4)
    NotebookClient(notebook, timeout=120, kernel_name='pulsesplit-submission',
                   resources={'metadata': {'path': str(destination)}}).execute()
    nbformat.write(notebook, notebook_path)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
    if any(cell.execution_count is None or any(item.output_type == 'error' for item in cell.outputs)
           for cell in code_cells):
        raise ValueError('Notebook execution did not complete')
    verification = {'matched_triplets': len(records), 'wav_files': 3 * len(records),
                    'executed_code_cells': len(code_cells), 'notebook_errors': 0,
                    'notebook_uses_saved_test_scores': True, 'validation_demo_only': True}
    (results / 'submission_checks.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    files = [path for path in destination.rglob('*') if path.is_file() and '__pycache__' not in path.parts]
    manifest = {path.relative_to(destination).as_posix(): file_sha256(path) for path in files}
    (results / 'SUBMISSION_SHA256.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    files.append(results / 'SUBMISSION_SHA256.json')
    archive = parent / 'PulseSplit_Submission.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(files):
            bundle.write(path, f'PulseSplit_Submission/{path.relative_to(destination).as_posix()}')
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError('ZIP validation failed')
        for name, digest in manifest.items():
            import hashlib
            if hashlib.sha256(bundle.read(f'PulseSplit_Submission/{name}')).hexdigest() != digest:
                raise ValueError(f'ZIP hash mismatch: {name}')
    print(json.dumps(verification, indent=2))
    print(f'Submission folder: {destination}')
    print(f'Send this ZIP: {archive}')


if __name__ == '__main__':
    main()
