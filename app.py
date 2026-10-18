"""Streamlit demo: upload a 15s or shorter clinical-manikin-style mixed WAV."""
from pathlib import Path
import io
import numpy as np
import soundfile as sf
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
from math import gcd
from pulsesplit.pipeline import load_checkpoint, separate_signal

st.set_page_config(page_title='PulseSplit | Heart + Lung',layout='wide')
st.title('PulseSplit')
st.caption('Experimental heart/lung audio separator — not for medical diagnosis or patient use')
checkpoint=Path('artifacts/best.pt')
if not checkpoint.exists():
    st.warning('No trained model found. Run: pulsesplit train, then pulsesplit evaluate')
    st.stop()
@st.cache_resource
def get_model(path):return load_checkpoint(path)
model,settings=get_model(checkpoint)
upload=st.file_uploader('Upload a mixed mono/stereo WAV file',type=['wav'])
if upload:
    audio,sr=sf.read(io.BytesIO(upload.read()),dtype='float32',always_2d=True)
    mono=audio.mean(axis=1)
    if sr!=settings['sample_rate']:
        g=gcd(int(sr),int(settings['sample_rate']))
        mono=resample_poly(mono,int(settings['sample_rate'])//g,int(sr)//g).astype(np.float32)
    if len(mono)>30*settings['sample_rate']:
        st.error('Limit audio to 30 seconds for this demo.');st.stop()
    scale=max(np.max(np.abs(mono)),1e-6)
    mixed=mono/scale
    predicted=separate_signal(model,mixed,seconds=settings['seconds'])
    names=['Mixed audio','Estimated heart audio','Estimated lung audio']
    signals=[mixed,predicted[0],predicted[1]]
    cols=st.columns(3)
    for col,name,sig in zip(cols,names,signals):
        with col:
            st.subheader(name)
            buffer=io.BytesIO();sf.write(buffer,sig,settings['sample_rate'],format='WAV')
            st.audio(buffer.getvalue(),format='audio/wav')
            fig,ax=plt.subplots(figsize=(5,2))
            ax.specgram(sig,Fs=settings['sample_rate'],NFFT=512,noverlap=384,cmap='magma')
            ax.set_ylim(0,1200);ax.set_xlabel('Seconds');ax.set_ylabel('Hz');fig.tight_layout()
            st.pyplot(fig);plt.close(fig)
st.info('Only validated on the dataset used for training. Per-recording test evaluation is in results/summary.json after training.')
