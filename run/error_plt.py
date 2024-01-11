import argparse
import os, sys
import collections
import numpy as np
import pandas as pd

from core.names import PATH_SPLIT
from tools.plot import lineplot_dataframe
from tools.utils import to_int

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('source',
            type=str)
  parser.add_argument('--prefix', '-p', 
            type=str, 
            default=None)
  parser.add_argument('--outdir', '-o', 
            type=str, 
            default='result')
  args = parser.parse_args()

  return args


def diff(df1, df2, col, mean=True):
  val = df1[col].to_numpy() - df2[col].to_numpy()
  if mean:
    val = np.mean(val)
  return val


LEGEND = 'legend'
COLUMN_NAME = 'abs error'


if __name__ == '__main__':
  args = parse_args()

  diff_errors = collections.defaultdict(collections.defaultdict(list))
  for sd in os.listdir(args.source):
    if not sd.startswith('seed'):
      continue
    for f in os.listdir(os.path.join(args.source, sd)):
      if args.prefix and not f.startswith(args.prefix):
        continue
      filepath = os.path.join(args.source, sd, f)
      df: pd.DataFrame = pd.read_csv(filepath)

      train_df = df.loc[df[LEGEND] == 'train']
      ego_df = df.loc[df[LEGEND] == 'ego']

      cat, step = f.split('.')[0].split('-')
      step = to_int(step)
      diff_errors[cat][step].append(diff(ego_df, train_df, COLUMN_NAME))

  x = 'steps'
  y = 'absolute error difference'
  data = collections.defaultdict(collections.defaultdict(list))
  for k, v in diff_errors.items():
    data = collections.defaultdict(list)
    for step in sorted(v.keys()):
      data[x].append(step)
      data[y].append(np.mean(diff_errors[step]))
    

  filename = args.path.rsplit(PATH_SPLIT, 1)[-1]
  filename = filename.split('.')[0]

  y = 'absolute error difference'
  data = {}
  data['steps'] = df['steps'].to_numpy()
  data[y] = diff(ego_df, train_df, COLUMN_NAME)
  data['legend'] = ['ego&train'] * data[y].size
  
  if 'lka' in df.columns:
    lka_df = df.loc[df[LEGEND] == 'lka']
    dlt = diff(lka_df, train_df, COLUMN_NAME)
    data['legend'] += ['lka&train'] * dlt.size()
  data = pd.DataFrame.from_dict(data=data)
  # print(data)
  lineplot_dataframe(data, title=f'error_diff_{filename}', y=y, outdir=args.outdir)
