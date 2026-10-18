"""Run reproducible software checks and package a teacher-facing submission.

Run from the project root: python tools/build_submission.py
Synthetic signals below are software fixtures, never UCI experiment results.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import platform
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    import numpy as np
    import soundfile as sf
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pulsesplit.pipeline import train, load_checkpoint, separate_signal, evaluate
    from pulsesplit.audit import audit_dataset

    # Each run gets its own directory so failed reruns cannot reuse old evidence.
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'submission' / stamp
    proof = output / 'proof'
    proof.mkdir(parents=True)
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', f'--junitxml={proof / "tests.xml"}'],
        cwd=ROOT, capture_output=True, text=True,
    )
    (proof / 'pytest.txt').write_text(result.stdout + result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'Tests failed; see {proof / "pytest.txt"}. No ZIP created.')

    torch.set_num_threads(1)
    sr = 8000
    time = np.arange(sr, dtype=np.float32) / sr
    log = io.StringIO()
    with TemporaryDirectory(prefix='pulsesplit-check-') as temporary:
        data = Path(temporary) / 'fixture'
        data.mkdir()
        for index in range(1, 10):
            heart = .2 * np.sin(2 * np.pi * (60 + index) * time)
            lung = .1 * np.sin(2 * np.pi * (180 + index) * time)
            for kind, signal in [('M', heart + lung), ('H', heart), ('L', lung)]:
                sf.write(data / f'{kind}{index:04d}.wav', signal, sr)
        with redirect_stdout(log):
            checkpoint = train(data, Path(temporary) / 'model', epochs=2,
                               batch_size=2, base=4, seconds=.5,
                               dataset_label='Synthetic software verification fixtures')
        audit = audit_dataset(data, proof / 'audit', splits=Path(checkpoint).parent / 'split.json')
        if audit['status'] != 'passed':
            raise RuntimeError('Synthetic fixture audit failed; no submission created')
        evaluate(data, checkpoint, proof / 'synthetic_evaluation')
        model, settings = load_checkpoint(checkpoint)
        mixed = (heart + lung).astype(np.float32)
        predicted = separate_signal(model, mixed, seconds=settings['seconds'])
        if predicted.shape != (2, sr) or not np.isfinite(predicted).all():
            raise RuntimeError('Checkpoint inference produced invalid audio')
        history = json.loads((Path(checkpoint).parent / 'history.json').read_text())
        if not all(np.isfinite(row[key]) for row in history
                   for key in ('train_logmag_mae', 'val_logmag_mae')):
            raise RuntimeError('Training produced a non-finite loss')
        for filename in ('history.json', 'run_metadata.json', 'split.json', 'training_history.png'):
            (proof / filename).write_bytes((Path(checkpoint).parent / filename).read_bytes())

    (proof / 'training_log.txt').write_text(log.getvalue(), encoding='utf-8')
    fig, axes = plt.subplots(2, 1, figsize=(9, 6))
    for key, label in [('train_logmag_mae', 'Train'), ('val_logmag_mae', 'Validation')]:
        axes[0].plot([r['epoch'] for r in history], [r[key] for r in history], 'o-', label=label)
    axes[0].set(xlabel='Epoch', ylabel='Log-magnitude L1 loss')
    axes[0].legend()
    for signal, label in [(mixed, 'Synthetic mixture'), (predicted[0], 'Heart estimate'),
                          (predicted[1], 'Lung estimate')]:
        axes[1].plot(time[:800], signal[:800], label=label, alpha=.75)
    axes[1].set(xlabel='Seconds', ylabel='Amplitude')
    axes[1].legend()
    fig.suptitle('SYNTHETIC SOFTWARE CHECK ONLY — not UCI performance')
    fig.tight_layout()
    fig.savefig(proof / 'software_check.png', dpi=160)
    plt.close(fig)
    environment = {'created_utc': stamp, 'python': sys.version,
                   'platform': platform.platform(),
                   'packages': {name: importlib.metadata.version(name)
                                for name in ('torch', 'numpy', 'scipy', 'pytest', 'soundfile')},
                   'checks': {'pytest': 'passed', 'two_epoch_training': 'passed',
                              'checkpoint_reload_and_inference': 'passed',
                              'dataset_audit': 'passed', 'evaluation_exports': 'passed'},
                   'data': 'Nine synthetic sine-wave triplets; NO real UCI experiment',
                   'inference_shape': list(predicted.shape)}
    (proof / 'verification.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    spec = importlib.util.spec_from_file_location('verification_report', ROOT / 'docs' / 'build_verification_report.py')
    reports = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reports)
    reports.build_reports(proof)
    (proof / 'READ_ME_FIRST.md').write_text(
        '# PulseSplit software verification\n\n'
        'These files prove that unit tests, two training epochs, checkpoint saving/loading, '
        'and audio inference executed successfully on synthetic fixtures.\n\n'
        'Open software_check.png for the visual evidence, pytest.txt/tests.xml for tests, '
        'training_log.txt/history.json for training, and verification.json for the environment.\n\n'
        'Software_Verification_Report.docx is an editable appendix with exact measured values. '
        'audit/ contains fixture quality checks; synthetic_evaluation/ exercises the evaluation '
        'exports. Every value in that folder is SYNTHETIC and cannot support real UCI performance claims.\n\n'
        '**This is not real UCI training, scientific performance evidence, or clinical validation.** '
        'The small checkpoint is temporary and is deliberately excluded to avoid using a fixture '
        'model as the real demo model. Run the README UCI commands for the actual experiment.\n',
        encoding='utf-8',
    )
    # Allowlist project files; never sweep up environments, credentials or raw data.
    files = [ROOT / name for name in ('README.md', 'LICENSE', 'pyproject.toml', 'app.py', '.gitignore')]
    for folder in ('src', 'tests', 'tools', 'docs', 'figures', 'notebooks', '.github'):
        files.extend(p for p in (ROOT / folder).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts
                     and p.suffix.lower() in {'.py', '.md', '.toml', '.yml', '.yaml', '.png',
                                              '.pdf', '.docx', '.ipynb'})
    files.extend(path for path in proof.rglob('*') if path.is_file())
    archive = output / 'PulseSplit_Submission.zip'
    hashes = {}
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(files):
            relative = Path('proof') / path.relative_to(proof) if path.is_relative_to(proof) else path.relative_to(ROOT)
            name = f'PulseSplit/{relative.as_posix()}'
            bundle.write(path, name)
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        bundle.writestr('PulseSplit/SHA256SUMS.json', json.dumps(hashes, indent=2))
    print(result.stdout)
    print('Software verification passed. No UCI performance claims.')
    print(f'Send this ZIP: {archive}')


if __name__ == '__main__':
    main()
