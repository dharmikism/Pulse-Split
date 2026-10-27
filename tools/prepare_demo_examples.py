"""Prepare three evenly spaced held-out examples for the local frontend."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pulsesplit.data import find_triplets
from pulsesplit.demo import prepare_upload, wav_bytes
from pulsesplit.pipeline import load_checkpoint, separate_signal


def main():
    checkpoint = ROOT / 'artifacts' / 'best.pt'
    model, settings = load_checkpoint(checkpoint)
    ids = sorted(json.loads((ROOT / 'artifacts' / 'split.json').read_text())['test'])
    chosen = [ids[index] for index in sorted({0, len(ids) // 2, len(ids) - 1})]
    triplets = {item.key: item for item in find_triplets(ROOT / 'data' / 'raw')}
    destination = ROOT / 'examples'
    destination.mkdir(exist_ok=True)
    manifest = []
    for index, key in enumerate(chosen, 1):
        mixed, metadata = prepare_upload(triplets[key].mixed.read_bytes(), target_rate=settings['sample_rate'])
        predictions = separate_signal(model, mixed, sample_rate=settings['sample_rate'], seconds=settings['seconds'])
        entry = {'id': key, 'title': f'Example {index} / Recording {key}',
                 'description': '15-second held-out UCI manikin recording. Compare the mixture with saved model estimates.',
                 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                 'sample_rate': metadata['sample_rate']}
        for role, signal in zip(('mixture', 'heart', 'lung'), (mixed, *predictions)):
            entry[role] = f'{key}_{role}.wav'
            (destination / entry[role]).write_bytes(wav_bytes(signal, metadata['sample_rate']))
        manifest.append(entry)
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Saved {len(manifest)} examples to {destination}')


if __name__ == '__main__':
    main()
