"""Create a portable notebook for the teacher submission layout."""
import json
from pathlib import Path


def build_notebook(destination):
    cells = []

    def markdown(source):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': source.splitlines(keepends=True)})

    def code(source):
        cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                      'outputs': [], 'source': source.splitlines(keepends=True)})

    markdown('# PulseSplit — Actual UCI Dataset Experiment\n\n'
             'This notebook reads the completed experiment and demonstrates the saved model. '
             'It does **not** retrain the model or recompute the held-out test scores.\n\n'
             '**Dataset:** [UCI HLS-CMDS](https://archive.ics.uci.edu/dataset/1202/hls-cmds), '
             'Torabi, Shirani and Reilly (2025), CC BY 4.0. The dataset folder contains '
             'the 435 WAVs in 145 matched triplets used by this experiment.\n\n'
             'The supplied outputs are from real data. Scores are SI-SDR in dB, not '
             'percentage accuracy. Separately recorded references have unverified '
             'time/phase alignment; these results do not establish clinical usefulness.')
    code("from pathlib import Path\nimport sys, json, hashlib\nimport numpy as np\nimport pandas as pd\nimport torch\nfrom IPython.display import display, Image, Audio\n\nROOT = Path.cwd()\nif not (ROOT / 'output').is_dir():\n    raise FileNotFoundError('Open this notebook from the extracted submission folder')\nMODELS = ROOT / 'output' / 'models'\nRESULTS = ROOT / 'output' / 'results'\nCHARTS = ROOT / 'output' / 'charts'\nsys.path.insert(0, str(MODELS / 'source'))\nsys.dont_write_bytecode = True\ntorch.set_num_threads(4)\nprint('Submission:', ROOT.name)\nprint('Python:', sys.version.split()[0])\nprint('PyTorch:', torch.__version__)\n")
    markdown('## 1. Verify the saved model and split\n\n'
             'Recordings sharing byte-identical audio are kept in one partition. '
             'Related simulator patterns can still occur across partitions.')
    code("def sha256(path):\n    return hashlib.sha256(path.read_bytes()).hexdigest()\n\nsummary = json.loads((RESULTS / 'summary.json').read_text())\nmetadata = json.loads((MODELS / 'run_metadata.json').read_text())\nsplit = json.loads((MODELS / 'split.json').read_text())\nassert sha256(MODELS / 'best.pt') == summary['checkpoint_sha256']\nassert sha256(MODELS / 'split.json') == summary['split_sha256']\nseen = set()\nhash_partitions = {}\nfor partition in ['train', 'val', 'test']:\n    ids = split[partition]\n    assert not seen.intersection(ids)\n    seen.update(ids)\n    for key in ids:\n        for digest in split['file_hashes'][key].values():\n            assert hash_partitions.get(digest, partition) == partition\n            hash_partitions[digest] = partition\nprint('Verified model and split hashes; no identical audio crosses partitions.')\ndisplay(pd.DataFrame([{'Partition': name, 'Recordings': len(split[name]),\n                       'Source groups': sum(v == name for v in split['group_assignments'].values())}\n                      for name in ['train', 'val', 'test']]))\ndisplay(pd.DataFrame([{'Epochs completed': metadata['epochs_completed'],\n                       'Selected epoch': metadata['selected_epoch'],\n                       'Parameters': metadata['parameter_count'],\n                       'Validation log-magnitude MAE': metadata['best_val_logmag_mae'],\n                       'Training seconds': metadata['runtime_seconds']}]))\n")
    markdown('## 2. Measured held-out results\n\n'
             'The following numbers come directly from the saved 22-recording evaluation. '
             'Positive gain means a higher score than using the mixture as the prediction. '
             'Both descriptive gain intervals include zero. Recording bootstrap intervals '
             'do not account for dependence within source groups.')
    code("rows = []\nfor source in ['heart', 'lung']:\n    gain = summary['statistics'][f'{source}_improvement']\n    rows.append({'Source': source.title(),\n                 'Mixture SI-SDR (dB)': summary['metrics'][f'{source}_baseline_si_sdr'],\n                 'Estimate SI-SDR (dB)': summary['metrics'][f'{source}_si_sdr'],\n                 'Mean gain (dB)': gain['mean'],\n                 'Descriptive gain CI': gain['mean_ci95']})\ndisplay(pd.DataFrame(rows).round(3))\nper_recording = pd.read_csv(RESULTS / 'test_per_recording.csv', dtype={'id': str})\nassert set(per_recording['id']) == set(split['test'])\nassert len(per_recording) == summary['num_test_recordings']\ndisplay(per_recording.round(3))\n")
    markdown('## 3. Training and evaluation charts')
    code("display(Image(filename=str(CHARTS / 'training_history.png')))\ndisplay(Image(filename=str(CHARTS / 'test_comparison.png')))\n")
    markdown('## 4. Dataset inspection\n\n'
             'The downloaded matched WAVs are mono, 4,000 Hz, and 15 seconds. '
             'This differs from the catalogue description of 22,050 Hz. Processing '
             'at 8,000 Hz adds no original recorded bandwidth. Near-full-scale '
             'samples are flagged as possible clipping, not proven distortion.')
    code("audit = json.loads((RESULTS / 'audit' / 'dataset_audit.json').read_text())\nprint('Matched triplets:', audit['recording_count'])\nprint('Matched WAVs:', audit['file_count'])\nprint('Native sample rates:', audit['sample_rates'])\nprint('Audit status:', audit['status'])\nprint('Warnings:', audit['issue_counts'])\nindex = pd.read_csv(ROOT / 'dataset' / 'recordings.csv', dtype={'id': str})\nassert set(index['id']) == seen\nprint('Dataset index verified:', len(index), 'matched triplets')\ndisplay(index.head())\n")
    markdown('## 5. Listen to a validation recording\n\n'
             'This example uses a validation recording, rather than selecting a test '
             'example by its score. Outputs are estimates from the actual trained model. '
             'Playback keeps the shared normalization and does not provide a new quality score.')
    code("from pulsesplit.audio import load_audio\nfrom pulsesplit.pipeline import load_checkpoint, separate_signal\n\nmodel, settings = load_checkpoint(MODELS / 'best.pt')\nexample_id = split['val'][0]\nexample_path = ROOT / 'dataset' / 'mixed' / f'M{example_id}.wav'\nmix = load_audio(example_path, settings['sample_rate'])\nmix = mix / max(float(np.max(np.abs(mix))), 1e-6)\npredicted = separate_signal(model, mix, sample_rate=settings['sample_rate'],\n                            seconds=settings['seconds'])\nassert predicted.shape == (2, len(mix)) and np.isfinite(predicted).all()\nprint('Validation recording:', example_id)\nfor label, signal in [('Mixture', mix), ('Estimated heart', predicted[0]), ('Estimated lung', predicted[1])]:\n    print(label)\n    display(Audio(signal, rate=settings['sample_rate'], normalize=False))\n")
    markdown('## Files to read\n\n'
             '- `output/results/Actual_Experiment_Report.docx`: editable report with measured results and limitations.\n'
             '- `output/results/test_per_recording.csv`: all held-out scores.\n'
             '- `output/models/best.pt`: selected trained checkpoint.\n'
             '- `output/models/split.json`: partition IDs, audio hashes, and source groups.\n'
             '- `output/charts/`: training and evaluation graphs.\n\n'
             'The original synthetic software checks are separate from this actual-data submission.')
    for index, cell in enumerate(cells):
        cell['id'] = f'pulsesplit-{index:02d}'
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3',
                 'language': 'python', 'name': 'python3'},
                 'language_info': {'name': 'python', 'version': sys_version()}},
                'nbformat': 4, 'nbformat_minor': 5}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding='utf-8')
    return destination


def sys_version():
    import platform
    return platform.python_version()
