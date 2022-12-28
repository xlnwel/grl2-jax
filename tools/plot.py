import os, sys, glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def plot_data_dict(data, *, x='step', outdir='results', figname='data'):
    fig = plt.figure(figsize=(40, 30))
    fig.tight_layout(pad=2)
    nrows = math.ceil(np.sqrt(len(data)))
    ncols = math.ceil(len(data) / nrows)

    keys  = sorted(list(data))
    for i, k in enumerate(keys):
        v = data[k]
        plot_data(v, x=x, y=k, outdir=outdir, title=k, 
            fig=fig, nrows=nrows, ncols=ncols, idx=i+1, 
            savefig=False
        )
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    fig.savefig(f'{outdir}/{figname}.png')

def plot_data(
    data, 
    *, 
    x='step', 
    y, 
    ys=None, 
    outdir='results', 
    title, 
    fig=None, 
    nrows=1, 
    ncols=1, 
    idx=1, 
    avg_data=False,
    savefig=True
):
    if isinstance(data, np.ndarray):
        seqlen = data.shape[-1]
        x_val = np.arange(seqlen)
        if data.ndim == 1:
            data = pd.DataFrame({x: x_val, y: data})
        elif data.ndim == 2:
            n = data.shape[0]
            if ys is not None:
                assert len(ys) == n, (len(ys), n)
                tag = [np.full((seqlen,), f'{yy}') for yy in ys]
            else:
                tag = [np.full((seqlen,), f'{y}{i}') for i in range(1, n+1)]
            assert len(tag) == n, (len(tag), n)
            tag = np.concatenate(tag)
            x_val = np.concatenate([x_val]*(n+1 if avg_data else n))
            y_val = np.concatenate(data)
            data = pd.DataFrame({x: x_val, 'tag': tag, y: y_val})
            data = data.pivot(x, 'tag', y)
            x = None
            y = None
        else:
            # return
            raise ValueError(f'Error data dimension: {data.shape}')
    assert isinstance(data, pd.DataFrame), type(data)
    
    if fig is None:
        fig = plt.figure(figsize=(20, 10))
        fig.tight_layout(pad=2)
    ax = fig.add_subplot(nrows, ncols, idx)
    sns.set(style="whitegrid", font_scale=1.5)
    sns.set_palette('Set2') # or husl
    sns.lineplot(x=x, y=y, ax=ax, data=data, dashes=False, linewidth=3)
    ax.grid(True, alpha=0.8, linestyle=':')
    ax.legend(loc='best').set_draggable(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title)
    if savefig:
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        outpath = f'{outdir}/{title}.png'
        fig.savefig(outpath)
        print(f'Plot Path: {outpath}')

def get_datasets(filedir, tag, condition=None):
    unit = 0
    datasets = []
    for root, _, files in os.walk(filedir):
        for f in files:
            if f.endswith('log.txt'):
                log_path = os.path.join(root, f)
                data = pd.read_csv(log_path, sep='\t')

                data.insert(len(data.columns), tag, condition)

                datasets.append(data)
                unit +=1

    return datasets

def event_to_pd(path_to_tb_event):
    print(path_to_tb_event)
    event_data = event_accumulator.EventAccumulator(path_to_tb_event)  # a python interface for loading Event data
    event_data.Reload()  # synchronously loads all of the data written so far b
    print('event tags', event_data.Tags())  # print all tags
    keys = event_data.scalars.Keys()  # get all tags,save in a list
    # print(keys)
    df = pd.DataFrame(columns=keys)  # my first column is training loss per iteration, so I abandon it
    # print(list(keys))
    if len(keys) == 0:
        return None
    for key in keys:
        df[key] = pd.DataFrame(event_data.Scalars(key)).value
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--title', '-t', default='', type=str)
    parser.add_argument('--legend', nargs='*')
    parser.add_argument('--legendtag', '-tag', default='Algo')
    parser.add_argument('--x', '-x', default='env_step', nargs='*')
    parser.add_argument('--y', '-y', default='score', nargs='*')
    parser.add_argument('--timing', default=None, choices=['Train', 'Eval', None], 
                        help='select timing to plot; both training and evaluation stats are plotted by default')
    args = parser.parse_args()

    # by default assume using `python utility/plot.py` to call this file
    if len(args.logdir) != 1:
        dirs = [f'{d}' for d in args.logdir]
    else:
        dirs = glob.glob(args.logdir[0])

    # dir follows pattern: logs/env/algo(/model_name)
    title = args.title or dirs[0].split('/')[1].split('_')[-1]
    # set up legends
    if args.legend:
        assert len(args.legend) == len(dirs), (
            "Must give a legend title for each set of experiments: "
            f"#legends({args.legend}) != #dirs({args.dirs})")
        legends = args.legend
    else:
        legends = [path.split('/')[2] for path in dirs]
        legends = [l[3:] if l.startswith('GS-') else l for l in legends]
    tag = args.legendtag

    print('Directories:')
    for d in dirs:
        print(f'\t{d}')
    print('Legends:')
    for l in legends:
        print(f'\t{l}')
    data = []
    for logdir, legend_title in zip(dirs, legends):
        data += get_datasets(logdir, tag, legend_title)

    xs = args.x if isinstance(args.x, list) else [args.x]
    ys = args.y if isinstance(args.y, list) else [args.y]
    for x in xs:
        for y in ys:
            outdir = f'results/{title}-{x}-{y}'
            plot_data(data, x, y, outdir, tag, title, args.timing)

if __name__ == '__main__':
    main()
