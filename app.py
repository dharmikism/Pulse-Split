"""Interactive heart/lung separation demo."""
from pathlib import Path
import hashlib
import io
import json
import zipfile
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pulsesplit.demo import prepare_upload, wav_bytes
from pulsesplit.pipeline import load_checkpoint, separate_signal

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='PulseSplit | Heart + Lung', layout='wide')
st.title('PulseSplit')
st.caption('Explore heart and lung sound separation · Research prototype')
st.info('Experimental estimates for project demonstrations. Not for diagnosis or patient use.')
with st.sidebar:
    st.header('How it works')
    st.markdown('1. Upload a mono or stereo WAV.\n2. Separate the recording.\n3. Listen, compare, and download.')
    st.caption('Up to 30 seconds · 20 MB maximum · processed at 8 kHz')
    view = st.radio('Visualization', ['Spectrogram', 'Waveform'])
    st.caption('Outputs are estimates. Assess quality using listening and held-out evaluation.')


@st.cache_resource
def get_model(path, modified_ns):
    return load_checkpoint(path)


@st.cache_data(max_entries=3, show_spinner=False)
def separate(payload, path, modified_ns):
    model, settings = get_model(path, modified_ns)
    mixed, metadata = prepare_upload(payload, target_rate=settings['sample_rate'])
    predictions = separate_signal(model, mixed, sample_rate=settings['sample_rate'], seconds=settings['seconds'])
    return mixed, predictions, metadata


checkpoint = ROOT / 'artifacts' / 'best.pt'
if not checkpoint.exists():
    st.warning('A trained model is required before separating recordings.')
    with st.expander('Set up the real dataset and model', expanded=True):
        st.code('pulsesplit download\npulsesplit inspect\npulsesplit train --epochs 30\npulsesplit evaluate', language='bash')
        st.write('Submission verification uses synthetic fixtures and does not install a demo model.')
    st.stop()

modified_ns = checkpoint.stat().st_mtime_ns
st.success('Model available · Ready to separate')
upload = st.file_uploader('Choose a mixed heart/lung recording', type=['wav'])
if upload is None:
    st.write('Upload a recording to compare the mixture with the estimated sources.')
    st.stop()
payload = upload.getvalue()
try:
    _, preview = prepare_upload(payload)
except ValueError as error:
    st.error(str(error))
    st.stop()
columns = st.columns(3)
columns[0].metric('Duration', f'{preview["duration_seconds"]:.2f} s')
columns[1].metric('Input sample rate', f'{preview["original_sample_rate"]:,} Hz')
columns[2].metric('Input channels', preview['channels'])
identity = (hashlib.sha256(payload).hexdigest(), modified_ns)
if st.button('Separate audio', type='primary'):
    with st.spinner('Estimating heart and lung signals…'):
        try:
            mixed, predictions, metadata = separate(payload, str(checkpoint), modified_ns)
            st.session_state['separation'] = (identity, mixed, predictions, metadata)
        except (ValueError, RuntimeError, KeyError, OSError) as error:
            st.error(f'Separation could not finish: {error}')
result = st.session_state.get('separation')
if result is None or result[0] != identity:
    st.stop()
_, mixed, predictions, metadata = result
st.caption('Shared mixture normalization · Float WAV downloads preserve output amplitudes.')
exports = {}
for column, label, filename, signal in zip(
    st.columns(3), ['Mixed audio', 'Estimated heart', 'Estimated lung'],
    ['mixture.wav', 'heart_estimate.wav', 'lung_estimate.wav'],
    [mixed, predictions[0], predictions[1]],
):
    with column:
        st.subheader(label)
        encoded = wav_bytes(signal, metadata['sample_rate'])
        exports[filename] = encoded
        st.audio(encoded, format='audio/wav')
        fig, axis = plt.subplots(figsize=(5, 2.5))
        if view == 'Waveform':
            axis.plot(np.arange(len(signal)) / metadata['sample_rate'], signal, linewidth=.6)
            axis.set_ylabel('Amplitude')
        else:
            padded = np.pad(signal, (0, max(0, 512 - len(signal))))
            axis.specgram(padded, Fs=metadata['sample_rate'], NFFT=512, noverlap=384, cmap='magma')
            axis.set_ylim(0, 1200)
            axis.set_ylabel('Hz')
        axis.set_xlabel('Seconds')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.download_button('Download WAV', encoded, file_name=filename, mime='audio/wav')
metadata['disclaimer'] = 'Experimental estimates; no clinical validation. No quality score is inferred from this upload.'
bundle = io.BytesIO()
with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, encoded in exports.items():
        archive.writestr(name, encoded)
    archive.writestr('recording_info.json', json.dumps(metadata, indent=2))
st.download_button('Download all audio + recording information', bundle.getvalue(),
                   file_name='PulseSplit_Audio.zip', mime='application/zip')
