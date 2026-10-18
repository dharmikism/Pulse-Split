from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
OUT=Path(__file__).resolve().parents[1]/'figures';OUT.mkdir(exist_ok=True)

def flow(filename,labels,title):
    fig,ax=plt.subplots(figsize=(9,3.1))
    ax.set_xlim(0,10);ax.set_ylim(0,3.2);ax.axis('off')
    n=len(labels);gap=.18;boxw=(9.6-gap*(n-1))/n
    for j,(head,sub) in enumerate(labels):
        x=.2+j*(boxw+gap)
        rect=FancyBboxPatch((x,1),boxw,1.25,boxstyle='round,pad=.06,rounding_size=.10',facecolor='#e9f2f9',edgecolor='#386895',linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x+boxw/2,1.72,head,ha='center',va='center',fontsize=10,fontweight='bold')
        ax.text(x+boxw/2,1.29,sub,ha='center',va='center',fontsize=8,wrap=True)
        if j<n-1:
            xx=x+boxw+.035
            ax.add_patch(FancyArrowPatch((xx,1.62),(xx+gap-.055,1.62),arrowstyle='-|>',mutation_scale=14,color='#386895',lw=1.5))
    ax.set_title(title,fontsize=13,color='#0b5394',pad=8)
    fig.tight_layout();fig.savefig(OUT/filename,dpi=190,bbox_inches='tight');plt.close(fig)
flow('system_architecture.png',[('UCI HLS-CMDS','145 triplets'),('Preprocessing','8 kHz, segment'),('STFT','complex spectrum'),('U-Net','2 masks'),('Reconstruct','heart / lung')],'PulseSplit system architecture')
flow('experiment_flow.png',[('Split IDs','70 / 15 / 15'),('Train','fit U-Net'),('Validate','early stopping'),('Freeze model','best checkpoint'),('Test','SI-SDR + gain')],'Leakage-aware model evaluation')
flow('model_layers.png',[('Input','1 x F x T'),('Encoder','16 / 32 / 64'),('Bridge','128 channels'),('Decoder','skip connections'),('Softmax','heart + lung')],'Spectrogram-mask U-Net design')
