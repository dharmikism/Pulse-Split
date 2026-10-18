"""Training, reproducibility and held-out evaluation. No synthetic performance claims."""
from __future__ import annotations
from pathlib import Path
import json
import random
import time
import platform
from datetime import datetime, timezone
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import find_triplets, make_splits, SegmentDataset
from .model import MaskUNet, spectrogram, loss_fn, reconstruct
from .audio import si_sdr
from .reporting import validate_splits, file_sha256, write_evaluation_report, plot_training_history


def setup_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def get_data(root,seed=42,seconds=2):
    triplets=find_triplets(root)
    splits=make_splits(triplets,seed=seed)
    by_id={t.key:t for t in triplets}
    datasets={kind:SegmentDataset([by_id[x] for x in splits[kind]],seconds=seconds,training=kind=='train') for kind in ('train','val','test')}
    return triplets,splits,datasets


def run_epoch(model,loader,optimizer=None,device='cpu'):
    train=optimizer is not None
    model.train(train)
    losses=[]
    for wave,_ in loader:
        wave=wave.to(device)
        stft=spectrogram(wave.reshape(-1,wave.size(-1))).reshape(wave.size(0),3,257,-1)
        mix,truth=stft[:,0],stft[:,1:]
        with torch.set_grad_enabled(train):
            mask=model(mix.abs().unsqueeze(1))
            loss=loss_fn(mask,mix,truth)
            if train:
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        losses.append((loss.item(),wave.size(0)))
    return sum(v*n for v,n in losses)/sum(n for _,n in losses)


def train(root='data/raw',output='artifacts',seed=42,epochs=30,batch_size=8,base=16,seconds=2,patience=7,
          dataset_label='Unverified local recordings'):
    if epochs < 1 or batch_size < 1 or base < 1 or not np.isfinite(seconds) or seconds < .064 or patience < 1:
        raise ValueError('Use positive epochs, batch size, base and patience; segment duration must be at least 0.064 seconds')
    started = time.perf_counter()
    setup_seed(seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    triplets,splits,datasets=get_data(root,seed,seconds)
    dest=Path(output); dest.mkdir(parents=True,exist_ok=True)
    (dest/'split.json').write_text(json.dumps(splits,indent=2))
    train_dl=DataLoader(datasets['train'],batch_size=batch_size,shuffle=True)
    val_dl=DataLoader(datasets['val'],batch_size=batch_size)
    model=MaskUNet(base).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    best=float('inf'); stale=0; history=[]
    for epoch in range(1,epochs+1):
        tl=run_epoch(model,train_dl,opt,device)
        vl=run_epoch(model,val_dl,device=device)
        if not np.isfinite(tl) or not np.isfinite(vl):
            raise ValueError('Training produced non-finite loss; audit the source audio')
        history.append({'epoch':epoch,'train_logmag_mae':tl,'val_logmag_mae':vl})
        print(f'epoch={epoch:03d} train={tl:.5f} val={vl:.5f} device={device}',flush=True)
        if vl < best-1e-5:
            best=vl;stale=0
            torch.save({'state_dict':model.state_dict(),'base':base,'sample_rate':8000,'n_fft':512,
                        'hop':128,'seconds':seconds,'seed':seed,'best_val_loss':best},dest/'best.pt')
        else:
            stale+=1
            if stale>=patience:break
    (dest/'history.json').write_text(json.dumps(history,indent=2))
    (dest/'run_metadata.json').write_text(json.dumps({'torch':torch.__version__,'numpy':np.__version__,
        'python':platform.python_version(),'platform':platform.platform(),
        'completed_utc':datetime.now(timezone.utc).isoformat(),'runtime_seconds':time.perf_counter()-started,
        'dataset_label':dataset_label,'epochs_completed':len(history),'batch_size':batch_size,
        'base_channels':base,'segment_seconds':seconds,'parameter_count':sum(p.numel() for p in model.parameters()),
        'device':device,'seed':seed,'recordings':len(triplets),'split_sizes':{k:len(splits[k]) for k in ('train','val','test')},
        'segment_lengths':{k:len(datasets[k]) for k in ('train','val','test')},'best_val_logmag_mae':best},indent=2))
    plot_training_history(history, dest)
    return str(dest/'best.pt')


def load_checkpoint(path,device='cpu'):
    ckpt=torch.load(path,map_location=device,weights_only=False)
    model=MaskUNet(int(ckpt['base'])).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model,ckpt


@torch.no_grad()
def separate_signal(model,mix,sample_rate=8000,seconds=2.0,device='cpu'):
    """Process mono audio in windows; this is offline, not causal streaming."""
    mix = np.asarray(mix, dtype=np.float32)
    if mix.ndim != 1 or not np.isfinite(mix).all():
        raise ValueError('Inference expects finite mono audio')
    if not np.isfinite(seconds) or seconds <= 0 or sample_rate <= 0 or sample_rate * seconds < 1:
        raise ValueError('Sample rate and window duration must produce at least one sample')
    length=int(sample_rate*seconds)
    outs=[]
    for start in range(0,len(mix),length):
        piece=np.asarray(mix[start:start+length],dtype=np.float32)
        orig=len(piece)
        if orig<512:
            piece=np.pad(piece,(0,512-orig))
        tensor=torch.from_numpy(piece).to(device)[None]
        spec=spectrogram(tensor)
        masks=model(spec.abs().unsqueeze(1))
        predicted=reconstruct(spec,masks,length=len(piece))
        outs.append(predicted[0,:,:orig].cpu().numpy())
    if not outs:
        return np.zeros((2,0),dtype=np.float32)
    return np.concatenate(outs,axis=1)


def evaluate(root='data/raw',checkpoint='artifacts/best.pt',output='results', dataset_label=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .audio import load_audio,align_by_length,shared_scale
    dest=Path(output);dest.mkdir(parents=True,exist_ok=True)
    model,ckpt=load_checkpoint(checkpoint)
    split_path=Path(checkpoint).parent/'split.json'
    if not split_path.exists():raise FileNotFoundError('Split manifest missing: do not rerandomize test set')
    splits=json.loads(split_path.read_text())
    all_recs={x.key:x for x in find_triplets(root)}
    validate_splits(splits, all_recs)
    metadata_path = Path(checkpoint).parent / 'run_metadata.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    label = dataset_label or metadata.get('dataset_label', 'Unverified local recordings')
    rows=[]
    for key in splits['test']:
        triplet=all_recs[key]
        mix,h,l=shared_scale(*align_by_length(*(load_audio(p,ckpt['sample_rate']) for p in (triplet.mixed,triplet.heart,triplet.lung))))
        pred=separate_signal(model,mix,sample_rate=ckpt['sample_rate'],seconds=ckpt['seconds'])
        score_h=si_sdr(h,pred[0]);score_l=si_sdr(l,pred[1])
        baseline_h=si_sdr(h,mix);baseline_l=si_sdr(l,mix)
        rows.append({'id':key,'heart_si_sdr':score_h,'lung_si_sdr':score_l,'heart_baseline_si_sdr':baseline_h,
                     'lung_baseline_si_sdr':baseline_l,'heart_improvement':score_h-baseline_h,
                     'lung_improvement':score_l-baseline_l})
    import pandas as pd
    frame=pd.DataFrame(rows)
    frame.to_csv(dest/'test_per_recording.csv',index=False)
    # Report both per-source metrics and relative gain over unprocessed mixture.
    summary={'num_test_recordings':len(rows),'dataset_label':label,
             'checkpoint_sha256':file_sha256(checkpoint),'split_sha256':file_sha256(split_path),
             'completed_utc':datetime.now(timezone.utc).isoformat(),
             'disclaimer':'Measured only on the supplied held-out recordings. Dataset label is user supplied, not independently verified. No clinical validation.'}
    write_evaluation_report(frame, summary, dest)
    fig,ax=plt.subplots(figsize=(8,4))
    metric_cols=['heart_baseline_si_sdr','heart_si_sdr','lung_baseline_si_sdr','lung_si_sdr']
    ax.bar(range(4),[summary['metrics'][c] if summary['metrics'][c] is not None else np.nan for c in metric_cols]);ax.set_xticks(range(4),['Heart\nmixture','Heart\nseparated','Lung\nmixture','Lung\nseparated'])
    ax.set_ylabel('Mean SI-SDR (dB)');ax.set_title('Held-out recordings | source vs unprocessed mixture')
    fig.suptitle(label)
    fig.tight_layout();fig.savefig(dest/'test_comparison.png',dpi=180);plt.close(fig)
    return summary
