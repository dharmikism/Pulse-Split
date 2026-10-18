import argparse
from pathlib import Path
from urllib.request import urlretrieve
from .data import unpack_dataset, find_triplets
from .pipeline import train, evaluate
from .audit import audit_dataset

# Download must be from UCI, not any unapproved alternative source.
UCI_ZIP='https://archive.ics.uci.edu/static/public/1202/hls-cmds%3A%2Bheart%2Band%2Blung%2Bsounds%2Bdataset%2Brecorded%2Bfrom%2Ba%2Bclinical%2Bmanikin%2Busing%2Bdigital%2Bstethoscope.zip'

def main():
    parser=argparse.ArgumentParser(prog='pulsesplit')
    subs=parser.add_subparsers(dest='command',required=True)
    d=subs.add_parser('download');d.add_argument('--root',default='data/raw');d.add_argument('--archive',default=None)
    chk=subs.add_parser('inspect');chk.add_argument('--root',default='data/raw')
    audit=subs.add_parser('audit',help='Export dataset quality checks before training')
    audit.add_argument('--root',default='data/raw');audit.add_argument('--output',default='results/audit')
    audit.add_argument('--splits',default=None,help='Optional split.json to check recording overlap')
    t=subs.add_parser('train');t.add_argument('--root',default='data/raw');t.add_argument('--output',default='artifacts');t.add_argument('--epochs',type=int,default=30);t.add_argument('--batch-size',type=int,default=8);t.add_argument('--base',type=int,default=16);t.add_argument('--seconds',type=float,default=2.);t.add_argument('--seed',type=int,default=42)
    e=subs.add_parser('evaluate');e.add_argument('--root',default='data/raw');e.add_argument('--checkpoint',default='artifacts/best.pt');e.add_argument('--output',default='results')
    t.add_argument('--dataset-label',default='Unverified local recordings')
    e.add_argument('--dataset-label',default=None,help='Defaults to the training run data label')
    args=parser.parse_args()
    if args.command=='download':
        root=Path(args.root);root.mkdir(parents=True,exist_ok=True)
        archive=Path(args.archive) if args.archive else root/'hls-cmds-original.zip'
        if not archive.exists():
            if args.archive:raise FileNotFoundError(archive)
            print('Downloading original UCI data ...')
            urlretrieve(UCI_ZIP,archive)
        print(f'Extracting {archive} ...')
        unpack_dataset(archive,root)
        print(f'Paired recordings discovered: {len(find_triplets(root))}')
    elif args.command=='inspect':
        rec=find_triplets(args.root)
        print(f'Paired recordings: {len(rec)}')
        for item in rec[:4]:print(item)
    elif args.command=='train':
        print('Best checkpoint:', train(args.root,args.output,args.seed,args.epochs,args.batch_size,args.base,args.seconds,dataset_label=args.dataset_label))
    elif args.command=='audit':
        import json
        print(json.dumps(audit_dataset(args.root,args.output,splits=args.splits),indent=2))
    elif args.command=='evaluate':
        print(evaluate(args.root,args.checkpoint,args.output,dataset_label=args.dataset_label))

if __name__=='__main__':main()
