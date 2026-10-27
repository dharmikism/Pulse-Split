"""Interactive heart/lung separation demo."""
from pathlib import Path
import hashlib
import io
import json
import zipfile
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pulsesplit.demo import prepare_upload, wav_bytes, playback_bytes
from pulsesplit.pipeline import load_checkpoint, separate_signal

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='PulseSplit | Heart + Lung', layout='wide')
st.markdown("""<style>
.stApp {background: var(--background-color); color: var(--text-color);}
.block-container {max-width: 1280px; padding-top: 2.5rem;}
h1, h2, h3 {color: var(--text-color); letter-spacing: -.035em;}
[data-testid="stSidebar"] {background: var(--secondary-background-color);}
[data-testid="stMetric"] {background: var(--secondary-background-color);
 color: var(--text-color); border: 1px solid color-mix(in srgb, var(--text-color) 18%, transparent);
 border-radius: 14px; padding: 18px;}
[data-testid="stAudio"] {margin: 12px 0;}
.hero {background: #102c42; color: #fff; padding: 36px 40px;
 border-radius: 22px; margin-bottom: 24px;}
.hero h1 {color: white; font-size: 3rem; margin: 8px 0;}
.hero p {color: #c3d6e4; max-width: 650px; font-size: 1.08rem;}
.eyebrow {color: #68dfca; font-size: .75rem; letter-spacing: .18em; font-weight: 700;}
</style><div class="hero"><div class="eyebrow">PULSESPLIT / AUDIO RESEARCH LAB</div>
<h1>One recording. Two perspectives.</h1>
<p>Listen inside a cardiopulmonary recording. Explore the original mix alongside
estimated heart and lung sounds, with ready-to-play examples.</p></div>""", unsafe_allow_html=True)
st.caption('Explore heart and lung sound separation · Research prototype')
st.info('Experimental estimates for project demonstrations. Not for diagnosis or patient use.')
with st.sidebar:
    st.header('How it works')
    st.markdown('1. Choose a saved example or upload a WAV.\n2. Compare the mixture and estimates.\n3. Explore the signals and download audio.')
    st.caption('Up to 30 seconds · 20 MB maximum · processed at 8 kHz')
    view = st.radio('Visualization', ['Spectrogram', 'Waveform'], key='visualization')
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


examples_path = ROOT / 'examples' / 'manifest.json'
examples = json.loads(examples_path.read_text(encoding='utf-8')) if examples_path.exists() else []
st.subheader('Explore a recording')
mode = st.radio('Audio source', ['Saved examples', 'Upload WAV'], horizontal=True, key='audio_source')
selected = None
upload = None
if mode == 'Saved examples' and examples:
    selected = st.selectbox('Choose a stored example', examples, format_func=lambda item: item['title'])
    st.caption(selected['description'])
    payload = (ROOT / 'examples' / selected['mixture']).read_bytes()
else:
    if mode == 'Saved examples':
        st.info('No saved examples installed. You can still upload a recording.')
    upload = st.file_uploader('Choose a mixed heart/lung recording', type=['wav'])
    payload = upload.getvalue() if upload is not None else None

with st.expander('About the experiment and dataset'):
    st.write('A compact U-Net predicts two spectral masks from a single recording. '
             'Training uses UCI HLS-CMDS clinical manikin recordings, with source-group splits before segmentation.')
    summary_path = ROOT / 'results' / 'summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        metrics = summary['metrics']
        cols = st.columns(3)
        cols[0].metric('Held-out recordings', summary['num_test_recordings'])
        cols[1].metric('Mean heart SI-SDR gain', f"{metrics['heart_improvement']:+.3f} dB")
        cols[2].metric('Mean lung SI-SDR gain', f"{metrics['lung_improvement']:+.3f} dB")
        st.caption('Both gain intervals include zero. Reference timing and phase are unverified; '
                   'these measurements do not establish clinical usefulness.')
    st.markdown('Dataset: Torabi, Shirani & Reilly (2025), [UCI HLS-CMDS](https://archive.ics.uci.edu/dataset/1202/hls-cmds), CC BY 4.0. '
                'Saved examples are held-out manikin recordings, not patient recordings.')

checkpoint = ROOT / 'artifacts' / 'best.pt'
if selected is None and not checkpoint.exists():
    st.warning('A trained model is required before separating recordings.')
    with st.expander('Set up the real dataset and model', expanded=True):
        st.code('pulsesplit download\npulsesplit inspect\npulsesplit train --epochs 30\npulsesplit evaluate', language='bash')
        st.write('Submission verification uses synthetic fixtures and does not install a demo model.')
    st.stop()
if payload is None:
    st.info('Choose an example or upload a recording to begin.')
    st.stop()
modified_ns = checkpoint.stat().st_mtime_ns if checkpoint.exists() else 0

try:
    _, preview = prepare_upload(payload)
except ValueError as error:
    st.error(str(error))
    st.stop()
columns = st.columns(3)
columns[0].metric('Duration', f'{preview["duration_seconds"]:.2f} s')
columns[1].metric('Input sample rate', f'{preview["original_sample_rate"]:,} Hz')
columns[2].metric('Input channels', preview['channels'])
identity = (hashlib.sha256(payload).hexdigest(), modified_ns,
            selected['id'] if selected is not None else 'upload')
if selected is not None:
    import soundfile as sf
    mixed, metadata = prepare_upload(payload)
    predictions = np.stack([sf.read(ROOT / 'examples' / selected[name], dtype='float32')[0]
                            for name in ('heart', 'lung')])
    metadata['example_id'] = selected['id']
    metadata['checkpoint_sha256'] = selected['checkpoint_sha256']
    st.session_state['separation'] = (identity, mixed, predictions, metadata)
    st.success('Saved model estimates loaded. Listen and compare below.')
if selected is None and st.button('Separate audio', type='primary'):
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
st.subheader('Listen & compare')
boost = st.checkbox('Boost quiet audio for listening', value=True)
st.caption('Playback uses browser-compatible 16-bit audio. Boost adjusts each player independently; '
           'plots and downloads retain the original amplitudes. Headphones help with low-frequency heart sounds.')
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
        st.audio(playback_bytes(signal, metadata['sample_rate'], boost=boost), format='audio/wav')
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
