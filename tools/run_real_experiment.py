"""Audit, train and evaluate downloaded UCI recordings, preserving run evidence.

First run `pulsesplit download`, then `python tools/run_real_experiment.py`.
This script never substitutes synthetic audio for an unavailable real dataset.
"""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import torch

from pulsesplit.audit import audit_dataset
from pulsesplit.cli import UCI_ZIP
from pulsesplit.data import find_triplets
from pulsesplit.grouping import make_source_group_splits
from pulsesplit.pipeline import train, evaluate
from pulsesplit.reporting import file_sha256

ROOT = Path(__file__).resolve().parents[1]


class Tee:
    def __init__(self, terminal, file):
        self.terminal, self.file = terminal, file

    def write(self, text):
        self.terminal.write(text)
        self.file.write(text)
        self.file.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.file.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'data' / 'raw')
    parser.add_argument('--artifacts', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    if (args.artifacts / 'best.pt').exists() or (args.results / 'summary.json').exists():
        parser.error('A model or completed evaluation already exists. Choose new output directories to preserve it.')
    torch.set_num_threads(args.threads)
    triplets = find_triplets(args.root)
    if len(triplets) != 145:
        raise ValueError(f'Expected all 145 official matched triplets; found {len(triplets)}')
    splits = make_source_group_splits(triplets, seed=42)
    audit = audit_dataset(args.root, args.results / 'audit', splits=splits)
    if audit['status'] == 'failed':
        raise ValueError('Dataset audit failed; review results/audit before training')
    args.results.mkdir(parents=True, exist_ok=True)
    archive = args.root / 'hls-cmds-original.zip'
    provenance = {
        'dataset': 'UCI HLS-CMDS',
        'dataset_page': 'https://archive.ics.uci.edu/dataset/1202/hls-cmds',
        'download_url': UCI_ZIP,
        'license': 'CC BY 4.0',
        'attribution': 'Torabi, Y., Shirani, S., & Reilly, J. (2025). HLS-CMDS. UCI Machine Learning Repository. DOI: 10.1109/IEEEDATA.2025.3566012.',
        'archive_sha256': file_sha256(archive) if archive.exists() else None,
        'archive_bytes': archive.stat().st_size if archive.exists() else None,
        'recording_triplets': len(triplets),
        'wav_count': sum(p.is_file() and p.suffix.lower() == '.wav' and not p.name.startswith('._')
                         and '__MACOSX' not in p.parts for p in args.root.rglob('*')),
        'seed': 42, 'threads': args.threads,
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'audit_status': audit['status'],
        'native_sample_rates': audit['sample_rates'],
        'audit_issue_counts': audit['issue_counts'],
        'split_strategy': splits['strategy'],
        'scope': 'Exact-audio connected-component holdout; no byte-identical M/H/L source crosses partitions. Similar simulator sound patterns can still repeat. Reference tracks were recorded separately, with unverified sample alignment. No clinical validation.',
    }
    (args.results / 'dataset_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    (args.results / 'PRETRAINING_PROTOCOL.md').write_text(
        '# Actual-data protocol fixed before training and test evaluation\n\n'
        'Use all 145 official UCI HLS-CMDS matched triplets, seed 42. Group recording IDs '
        'by connected components of byte-identical M/H/L audio before assigning partitions. '
        'This replaces the original random-ID plan because official files reuse identical '
        'sources and contain a duplicate mixture. No model test outputs informed this change.\n\n'
        'Train a base-16 U-Net, 8 kHz processing, 2-second windows, batch size 8, AdamW '
        'learning rate 0.001, up to 30 epochs with validation patience 7. Select the lowest '
        'validation log-magnitude loss under the existing 1e-5 improvement rule. Run held-out '
        'test evaluation once after model selection. Do not tune to test scores.\n\n'
        'Actual native WAVs are mono 4 kHz / 15 seconds; the catalogue claims 22.05 kHz. '
        'Upsampling to 8 kHz adds no recorded bandwidth. Retain possible clipping warnings. '
        'Separately recorded references are not verified sample-aligned additive stems. '
        'Raw reference SI-SDR is phase-sensitive and is not clinical accuracy. Exact hash '
        'grouping does not establish independence of all simulator patterns.\n', encoding='utf-8')
    # Include per-file hashes for the exact 435 matched WAVs used by this run.
    manifest = {str(path.relative_to(args.root)): file_sha256(path)
                for rec in triplets for path in (rec.mixed, rec.heart, rec.lung)}
    (args.results / 'audio_sha256.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Official data: {len(triplets)} triplets; audit={audit["status"]}; split sizes={[(key, len(splits[key])) for key in ("train", "val", "test")]}', flush=True)
    with (args.results / 'training_log.txt').open('w', encoding='utf-8') as log:
        with redirect_stdout(Tee(sys.stdout, log)):
            checkpoint = train(args.root, args.artifacts, epochs=args.epochs, batch_size=8,
                               base=16, seconds=2., seed=42, patience=7, dataset_label='UCI HLS-CMDS',
                               split_manifest=splits)
    # No hyperparameter changes are made after this single final evaluation.
    summary = evaluate(args.root, checkpoint, args.results)
    spec = importlib.util.spec_from_file_location('experiment_report', ROOT / 'docs' / 'build_experiment_report.py')
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    report.build_experiment_report(args.artifacts, args.results, args.results / 'report')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = ROOT / 'submission' / f'{stamp}_uci'
    output.mkdir(parents=True)
    archive_path = output / 'PulseSplit_UCI_Results.zip'
    files = [p for folder in (args.artifacts, args.results) for p in folder.rglob('*')
             if p.is_file() and p.name != '.gitkeep' and '__pycache__' not in p.parts]
    checksums = {}
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(files):
            parent = args.artifacts if path.is_relative_to(args.artifacts) else args.results
            prefix = 'artifacts' if parent == args.artifacts else 'results'
            relative = f'PulseSplit_UCI/{prefix}/{path.relative_to(parent).as_posix()}'
            bundle.write(path, relative)
            checksums[relative] = file_sha256(path)
        bundle.writestr('PulseSplit_UCI/SHA256SUMS.json', json.dumps(checksums, indent=2))
        bundle.writestr('PulseSplit_UCI/READ_ME_FIRST.md',
                        '# Actual UCI HLS-CMDS experiment\n\n'
                        'Read results/report/Actual_Experiment_Report.docx or its Markdown equivalent for '
                        'the measured results and limitations. Dataset details and attribution are in '
                        'results/dataset_provenance.json. The model is in artifacts/best.pt.\n\n'
                        'This archive contains experiment evidence and a trained model. '
                        'Raw dataset audio and synthetic verification outputs are excluded. '
                        'Source code: https://github.com/dharmikism/Pulse-Split\n')
    print(json.dumps(summary['metrics'], indent=2), flush=True)
    print(f'Actual-data results ZIP: {archive_path}', flush=True)


if __name__ == '__main__':
    main()
