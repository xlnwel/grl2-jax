import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('metric',
            type=str,
            default='score')
  args = parser.parse_args()
  
  return args


def yield_dirs(directory, dirname, is_suffix=True, matches=None):
  if not os.path.exists(directory):
    return []
  directory = directory
  n_slashes = dirname.count('/')
  
  for root, _, _ in os.walk(directory):
    if 'src' in root:
      continue
    if matches is not None and all([m not in root for m in matches]):
      continue

    endnames = root.rsplit('/', n_slashes+1)[1:]
    endname = '/'.join(endnames)
    if is_suffix:
      if endname.endswith(dirname):
        yield root
    else:
      if endname.startswith(dirname):
        yield root


if __name__ == '__main__':
  args = parse_args()
  metric = args.metric
  print('Metric:', metric)

  record_name = 'record.txt'
  # results = []
  # for i in range(1, 11):
  #   directory1 = f'/System/Volumes/Data/mnt/公共区/cxw/magw-logs/magw-staghunt/zero/1209/n_lka_steps=0/seed={i}/a0/record.txt'
  #   directory2 = f'/System/Volumes/Data/mnt/公共区/cxw/magw-logs/magw-staghunt/zero/1209/n_lka_steps=0/seed={i}/a0/record.txt'

  #   data1 = pd.read_table(directory1, on_bad_lines='skip')
  #   data2 = pd.read_table(directory2, on_bad_lines='skip')
  #   print(list(data1) == list(data2))
  #   print(all(data1['metrics/old_old_score'] == data2['metrics/old_old_score']))
  
  # for i in range(1, 11):
  #   directory1 = f'/System/Volumes/Data/mnt/公共区/cxw/magw-logs/magw-staghunt/zero/1206/baseline/seed={i}/a0/record.txt'
  #   directory2 = f'/System/Volumes/Data/mnt/公共区/cxw/magw-logs/magw-staghunt/zero/1206/run_with_future_opponents=True-n_lka_steps=3/seed={i}/a0/record.txt'

  #   data1 = pd.read_table(directory1, on_bad_lines='skip')
  #   data2 = pd.read_table(directory2, on_bad_lines='skip')
  #   print(list(data1) == list(data2))
  #   print(all(data1['metrics/old_old_score'] == data2['metrics/old_old_score']))
  # print(all(data1['metrics/score'] == data2['metrics/score']))

  for env in ('staghunt', 'escalation'):
    for i in (0, 1, 3, 5):
      directory = f'/System/Volumes/Data/mnt/公共区/cxw/magw-logs/magw-{env}/zero/1209/n_lka_steps={i}/'
      results = []
      final = []
      for d in yield_dirs(directory, 'a0', is_suffix=False):
        record_path = '/'.join([d, record_name])

        data = pd.read_table(record_path, on_bad_lines='skip')
        res = data[f'metrics/old_old_{metric}']
        results.append(res)
        final.append(data[f'metrics/{metric}'].iloc[-1])
      results = np.mean(results, 0)
      # print(ppo_res)
      # ppo_res = np.sort(ppo_res)[::-1]
      if i == 0:
        name = 'PPO'
        blank1 = 3
        blank2 = 6
      else:
        name = f'Future Step={i}'
        blank1 = 1
        blank2 = 5
        
      print(f'{env}-{name}. {metric.capitalize()} difference before and after training:', 
        '\t'*blank1, f'{np.nanmean(results):4.4g}(std={np.nanstd(results):4.4g})')
      print(f'{env}-{name}. Final {metric}:', '\t'*blank2, f'{np.nanmean(final):4.4g}')


  # directory = '/System/Volumes/Data/mnt/公共区/cxw/magw-logs/magw-staghunt/zero/1206/run_with_future_opponents=True-n_lka_steps=3'
  # results = []
  # for d in yield_dirs(directory, 'a0', is_suffix=False):
  #   record_path = '/'.join([d, record_name])

  #   try:
  #     data = pd.read_table(record_path, on_bad_lines='skip')
  #   except:
  #     continue
  #   res = data[f'metrics/old_old_{metric}']
  #   results.append(res)
  # fut_res = np.mean(results, 0)
  # fut_res = np.sort(fut_res)[::-1]
  # print('3-Step Future Opponents. Performance difference before and after training:\t', np.mean(fut_res), np.std(fut_res))

  # print('No Apple')
  # directory = '/System/Volumes/Data/mnt/公共区/cxw/magw-logs/magw-staghunt/zero/1206/has_hare=False'
  # results = []
  # for d in yield_dirs(directory, 'a0', is_suffix=False):
  #   record_path = '/'.join([d, record_name])
    
  #   data = pd.read_table(record_path, on_bad_lines='skip')
  #   res = data[f'metrics/old_old_{metric}']
  #   results.append(res)
  # ppo2_res = np.mean(results, 0)
  # ppo2_res = np.sort(ppo2_res)[::-1]
  # print('PPO. Performance difference before and after training:', '\t'*4, np.mean(ppo2_res), np.std(ppo2_res))

  # directory = '/System/Volumes/Data/mnt/公共区/cxw/magw-logs/magw-staghunt/zero/1206/has_hare=False-run_with_future_opponents=True-n_lka_steps=3'
  # results = []
  # for d in yield_dirs(directory, 'a0', is_suffix=False):
  #   record_path = '/'.join([d, record_name])

  #   try:
  #     data = pd.read_table(record_path, on_bad_lines='skip')
  #   except:
  #     continue
  #   res = data[f'metrics/old_old_{metric}']
  #   results.append(res)
  # fut2_res = np.mean(results, 0)
  # fut2_res = np.sort(fut2_res)[::-1]
  # print('3-Step Future Opponents. Performance difference before and after training:\t', np.mean(fut2_res), np.std(fut2_res))

  # x = np.arange(len(ppo_res))
  # plt.plot(x, ppo_res, label='ppo')
  # plt.plot(x, fut_res, label='future')
  # plt.plot(x, ppo2_res, label='ppo-no_apple')
  # plt.plot(x, fut2_res, label='future-no_apple')
  # plt.xlabel('argsort')
  # plt.ylabel('score')
  # plt.legend()
  # plt.savefig('comp.png')
  # plt.cla()

  # bins = 20
  # r = (-5, 5)
  # counts, bins = np.histogram(ppo_res, bins=bins, range=r)
  # plt.stairs(counts, bins, label='ppo')
  # counts, bins = np.histogram(fut_res, bins=bins, range=r)
  # plt.stairs(counts, bins, label='future')
  # counts, bins = np.histogram(ppo2_res, bins=bins, range=r)
  # plt.stairs(counts, bins, label='ppo-no_apple')
  # counts, bins = np.histogram(fut2_res, bins=bins, range=r)
  # plt.stairs(counts, bins, label='future-no_apple')
  # plt.xlabel('score')
  # plt.ylabel('number')
  # plt.legend()
  # plt.savefig('hist.png')


  # directory = '/System/Volumes/Data/mnt/公共区/cxw/magw-logs/magw-staghunt/zero/1130/run_with_future_opponents=True-n_lka_steps=3'
  # results = []
  # for d in yield_dirs(directory, 'a0', is_suffix=False):
  #   record_path = '/'.join([d, record_name])
    
  #   data = pd.read_table(record_path, on_bad_lines='skip')
  #   res = data[f'metrics/after_old_old_{metric}'] - data[f'metrics/before_old_old_{metric}']
  #   results.append(res.sum())

  # print('3-Step Future Opponents. Performance difference before and after training:\t', np.mean(results))
