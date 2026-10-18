"""Compact 2-channel U-Net for STFT source mask prediction."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, a,b):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(a,b,3,padding=1), nn.BatchNorm2d(b),nn.LeakyReLU(.1),
                               nn.Conv2d(b,b,3,padding=1),nn.BatchNorm2d(b),nn.LeakyReLU(.1))
    def forward(self,x): return self.net(x)

class MaskUNet(nn.Module):
    def __init__(self,base=16):
        super().__init__()
        self.enc1=ConvBlock(1,base)
        self.enc2=ConvBlock(base,base*2)
        self.enc3=ConvBlock(base*2,base*4)
        self.bridge=ConvBlock(base*4,base*8)
        self.dec3=ConvBlock(base*12,base*4)
        self.dec2=ConvBlock(base*6,base*2)
        self.dec1=ConvBlock(base*3,base)
        self.out=nn.Conv2d(base,2,1)
    def forward(self,magnitude):
        # log-compressed magnitude; normalizes by mean energy per spectrogram.
        x=torch.log1p(magnitude)
        a=self.enc1(x); b=self.enc2(F.avg_pool2d(a,2,ceil_mode=True)); c=self.enc3(F.avg_pool2d(b,2,ceil_mode=True))
        z=self.bridge(F.avg_pool2d(c,2,ceil_mode=True))
        for up,skip,block in ((None,c,self.dec3),(None,b,self.dec2),(None,a,self.dec1)):
            z=block(torch.cat([F.interpolate(z,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip],dim=1))
        return torch.softmax(self.out(z),dim=1)


def spectrogram(x,n_fft=512,hop=128):
    window=torch.hann_window(n_fft,device=x.device,dtype=x.dtype)
    return torch.stft(x,n_fft=n_fft,hop_length=hop,window=window,return_complex=True)


def reconstruct(stft_mix,mask,n_fft=512,hop=128,length=None):
    window=torch.hann_window(n_fft,device=stft_mix.device,dtype=stft_mix.real.dtype)
    return torch.stack([torch.istft(stft_mix*mask[:,i],n_fft=n_fft,hop_length=hop,window=window,length=length)
                        for i in range(2)],dim=1)


def loss_fn(masks, mix_spec, target_spec):
    # Magnitude regression is less brittle than sample-aligned waveform loss if
    # separately recorded ground truth sources have minor phase offsets.
    predicted = masks * mix_spec.abs().unsqueeze(1)
    target = target_spec.abs()
    return F.l1_loss(torch.log1p(predicted),torch.log1p(target))
