from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from pulsesplit.audio import load_audio, shared_scale, si_sdr
from pulsesplit.data import find_triplets, make_splits, SegmentDataset
from pulsesplit.model import MaskUNet, spectrogram, reconstruct,loss_fn
from pulsesplit.pipeline import separate_signal


def make_fixture(root,n=9):
    sr=8000;t=np.arange(sr,dtype=np.float32)/sr
    for i in range(1,n+1):
        h=(.2*np.sin(2*np.pi*(60+i)*t)).astype('float32')
        l=(.11*np.sin(2*np.pi*(180+i)*t)).astype('float32')
        for prefix,sig in [('M',h+l),('H',h),('L',l)]:
            folder=root/prefix;folder.mkdir(parents=True,exist_ok=True)
            sf.write(folder/f'{prefix}{i:04d}.wav',sig,sr)
    return root


def test_triplets_and_splits(tmp_path):
    root=make_fixture(tmp_path)
    rec=find_triplets(root)
    assert len(rec)==9
    parts=make_splits(rec)
    assert len(set(parts['train'])|set(parts['val'])|set(parts['test']))==9
    assert all(not set(parts[a])&set(parts[b]) for a,b in [('train','val'),('train','test'),('val','test')])
    subset=SegmentDataset(rec[:2],seconds=.5)
    x,_=subset[0]
    assert x.shape==(3,4000)


def test_model_shapes_loss_and_reconstruction():
    model=MaskUNet(base=4)
    samples=torch.randn(1,3,4096)*.05
    s=spectrogram(samples.reshape(-1,4096)).reshape(1,3,257,-1)
    masks=model(s[:,0:1].abs())
    assert masks.shape[1:]==(2,*s.shape[2:])
    assert torch.allclose(masks.sum(1),torch.ones_like(masks.sum(1)),atol=1e-6)
    assert torch.isfinite(loss_fn(masks,s[:,0],s[:,1:]))
    result=reconstruct(s[:,0],masks,length=4096)
    assert result.shape==(1,2,4096)
    result2=separate_signal(model,np.random.randn(2000).astype('float32'),seconds=.5)
    assert result2.shape==(2,2000)


def test_shared_scale_and_si_sdr():
    a=np.array([.5,-.2],dtype='float32')
    m,h,l=shared_scale(a,a*.6,a*.4)
    assert np.allclose(m,h+l)
    assert si_sdr(a,a)>si_sdr(a,np.array([-.5,-.2]))
