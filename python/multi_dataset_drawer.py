import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from pprint import pprint
from fractions import Fraction
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

BETA = '\u03B2'

def classification_report_json_to_df(ml_experiments_dir: Path, k_list, b_list=None, l_list=None):
  df = pd.DataFrame()
  for k in k_list:
    if l_list:
      for l in l_list:
        with open(ml_experiments_dir/f'k{k}/l{l}/classification_report.json', 'r') as json_file:
          report = json.load(json_file)
          flat_json = pd.json_normalize(report)
          flat_json['k'] = k
          flat_json['l'] = l
        df = pd.concat([df, flat_json], ignore_index=True)
    else:
      for b in b_list:
        with open(ml_experiments_dir/f'k{k}/B({b})/classification_report.json', 'r') as json_file:
          report = json.load(json_file)
          flat_json = pd.json_normalize(report)
          flat_json['k'] = k
          flat_json['b'] = b
        df = pd.concat([df, flat_json], ignore_index=True)
  df.fillna(0, inplace=True)
  avg_strats = [key for key in report.keys() if key not in ['accuracy', 'train_record_counts']]
  return (df, avg_strats)


def calculate_xticks_2(k_list, b_list, l_list, width, strat_gap):
   x_ticks = []
   x = np.arange(len(b_list))
   x_ticks[0:len(b_list)] = x + float(len(k_list) - 1) * width / 2

   x = np.arange(len(b_list)+strat_gap, len(b_list) + len(l_list)+strat_gap)
   x_ticks[len(b_list):len(b_list)+len(l_list)] = x + float(len(k_list) - 1) * width / 2 -0.1
   
   return x_ticks
      

def ldiv_compare_bsample_2():
  datasets = [
      {'name': 'ACSIncome 1664500', 'output_base_path': '../data/output2/ACSIncome_USA_2018_binned_imbalanced_1664500/', 'minority_class': '[100000-inf[', 'minority_class_name': 'rich', 'l_list': [2.0, 2.5, 3.0]},
      {'name': 'ACSIncome 16645', 'output_base_path': '../data/output2/ACSIncome_USA_2018_binned_imbalanced_16645/', 'minority_class': '[100000-inf[', 'minority_class_name': 'rich',  'l_list': [2.0, 2.5, 3.0]},      
      {'name': 'ACSIncome 1664', 'output_base_path': '../data/output2/ACSIncome_USA_2018_binned_imbalanced_1664/', 'minority_class': '[100000-inf[', 'minority_class_name': 'rich',  'l_list': [2.0, 2.5, 3.0]},    
      {'name': 'Nursery', 'output_base_path': '../data/output2/nursery/', 'minority_class': 'very_recom', 'minority_class_name': 'very_recom',  'l_list': [2.0, 3.0, 4.0]},
      {'name': 'CMC', 'output_base_path': '../data/output2/cmc/', 'minority_class': '2', 'minority_class_name': 'long-term',  'l_list': [2.0, 2.5, 3.0]},
  ]
  
  strats = ['BSAMPLE_V2', 'lDiv']
  strat_names = ['bsample_s', 'l-diversity']
  sample_experiment_numbers = [3, 3]
  k_list = [10, 20, 50, 100]
  b_list = [0.5, 0.1, 0.01]
  fraction_b_values = [Fraction(value).limit_denominator() for value in b_list]
  metric = 'f1-score'
  width = 0.225
  strat_gap = 0.5
  capsize = 3
  vline_x_position = 3
  l_offset = 47
  # colors = {20: 'tab:blue', 50: 'tab:orange', 100: 'tab:green'}
  colors = {10: 'tab:blue', 20: 'tab:orange', 50:'tab:green', 100:'tab:red'}
  hatches = ['\\\\', '//']
  
  fig, axes = plt.subplots(len(datasets), 2, figsize=(15, 10), sharey=True, sharex=False)
  plt.subplots_adjust(hspace=0.5, wspace=0.05)

  for i, dataset in enumerate(datasets):
      sample_df_list = []
      columns = ['macro avg', dataset['minority_class']]
      col_names = ['macro avg', dataset['minority_class_name']]
      labels = []
      l_list = dataset['l_list']
      for label in fraction_b_values + l_list:
         labels.append(label)

      for t, sample_strat in enumerate(strats):
          output_base_path = Path(dataset['output_base_path']).resolve()
          ml_experiments_dir = output_base_path / f'{sample_strat}/ml_experiments/experiment_{sample_experiment_numbers[t]}'
          if t == 0:
              df, avg_strats = classification_report_json_to_df(ml_experiments_dir, k_list, b_list=b_list)
          else:
              df, avg_strats = classification_report_json_to_df(ml_experiments_dir, k_list, l_list=l_list)
          sample_df_list.append(df)

      for j, col in enumerate(columns):
          multiplier = 0
      
          non_masked_balanced_report_path = output_base_path / 'inputdataset/folds/ml_experiments/experiment_3/balanced/classification_report.json'
          with open(non_masked_balanced_report_path, 'r') as f:
              non_masked_balanced_report = json.load(f)
              y_val = non_masked_balanced_report[columns[j]]['mean_f1-score']
              axes[i, j].axhline(y=y_val, label='original dataset\n(balanced)', color='red', linestyle='--', zorder=0)       

          for r, (df, strat) in enumerate(zip(sample_df_list, strats)):
              for k_idx, k in enumerate(k_list):
                  if r == 0:
                      k_rows = df.query(f'k == {k} and b in @b_list').sort_values(by='b', ascending=False)
                      x = np.arange(len(b_list))
                  else:
                      k_rows = df.query(f'k == {k} and l in @l_list').sort_values(by='l', ascending=True)
                      x = np.arange(len(b_list)-strat_gap, len(b_list) + len(l_list)-strat_gap)
                  mean_list = k_rows[f'{col}.mean_{metric}'].tolist()
                  min_list = k_rows[f'{col}.min_{metric}'].tolist()
                  max_list = k_rows[f'{col}.max_{metric}'].tolist()

                  offset = width * multiplier
                  lower_errors = [mean - min_val for mean, min_val in zip(mean_list, min_list)]
                  upper_errors = [max_val - mean for mean, max_val in zip(mean_list, max_list)]
                  error_bars = [lower_errors, upper_errors]
                  axes[i, j].bar(x + offset, mean_list, width, label=f'k={k}', yerr=error_bars, capsize=capsize, color=colors[k], hatch=hatches[r], edgecolor='white')
                  multiplier += 1

              # multiplier += k_gap / width

         
          x_label = 'β' + ' '*l_offset + '\u2113'
          # x_label = 'β' + ' '*l_offset + 'l'
          axes[i, j].axvline(x=vline_x_position, color='black', linestyle='-', linewidth=1, zorder=0)
          axes[i, j].set_title(f"{dataset['name']} - {col_names[j]} F1-score")
          axes[i, j].set_xlabel(x_label, loc='left', labelpad=-11)
          axes[i, j].set_xticklabels(labels)
          x_ticks = calculate_xticks_2(k_list, fraction_b_values, l_list, width, strat_gap)
          axes[i, j].set_xticks(x_ticks)


  legend_elements = [
      Line2D([0], [0], color='red', lw=2, linestyle='--', label='original dataset\n(balanced)'),
      Patch(facecolor='tab:blue', edgecolor='black', label='k=10'),
      Patch(facecolor='tab:orange', edgecolor='black', label='k=20'),
      Patch(facecolor='tab:green', edgecolor='black', label='k=50'),
      Patch(facecolor='tab:red', edgecolor='black', label='k=100'),
      Patch(facecolor='white', hatch=hatches[0], edgecolor='black', label=strat_names[0]),
      Patch(facecolor='white', hatch=hatches[1], edgecolor='black', label=strat_names[1])
  ]

  fig.legend(handles=legend_elements, loc='upper right')
  plt.show()

def ldiv_compare_bsample():
  datasets = [
      {'name': 'ACSIncome 1664500', 'output_base_path': '../data/output2/ACSIncome_USA_2018_binned_imbalanced_1664500/', 'minority_class': '[100000-inf[', 'minority_class_name': 'rich', 'l_list': [2.0, 2.5, 3.0]},
      {'name': 'ACSIncome 16645', 'output_base_path': '../data/output2/ACSIncome_USA_2018_binned_imbalanced_16645/', 'minority_class': '[100000-inf[', 'minority_class_name': 'rich',  'l_list': [2.0, 2.5, 3.0]},
      {'name': 'Nursery', 'output_base_path': '../data/output2/nursery/', 'minority_class': 'very_recom', 'minority_class_name': 'very_recom',  'l_list': [2.0, 3.0, 4.0]},
      {'name': 'CMC', 'output_base_path': '../data/output2/cmc/', 'minority_class': '2', 'minority_class_name': 'long-term',  'l_list': [2.0, 2.5, 3.0]},
      {'name': 'ACSIncome 1664', 'output_base_path': '../data/output2/ACSIncome_USA_2018_binned_imbalanced_1664/', 'minority_class': '[100000-inf[', 'minority_class_name': 'rich',  'l_list': [2.0, 2.5, 3.0]},
  ]
  
  strats = ['BSAMPLE_V2', 'lDiv']
  strat_names = ['bsample_s', 'l-diversity']
  sample_experiment_numbers = [3, 3]
  k_list = [20, 50, 100]
  b_list = [0.5, 0.2, 0.01]
  fraction_b_values = [Fraction(value).limit_denominator() for value in b_list]
  metric = 'f1-score'
  width = 0.13
  k_gap = 0.03
  colors = {20: 'tab:blue', 50: 'tab:orange', 100: 'tab:green'}
  hatches = ['\\\\', '//']
  
  fig, axes = plt.subplots(len(datasets), 2, figsize=(15, 10), sharey=True, sharex=False)
  plt.subplots_adjust(hspace=0.5, wspace=0.05)

  for i, dataset in enumerate(datasets):
      sample_df_list = []
      columns = ['macro avg', dataset['minority_class']]
      col_names = ['macro avg', dataset['minority_class_name']]
      labels = []
      l_list = dataset['l_list']
      for b, l in zip(fraction_b_values, l_list):
         labels.append(f'{b}\n{l}')

      for t, sample_strat in enumerate(strats):
          output_base_path = Path(dataset['output_base_path']).resolve()
          ml_experiments_dir = output_base_path / f'{sample_strat}/ml_experiments/experiment_{sample_experiment_numbers[t]}'
          if t == 0:
              df, avg_strats = classification_report_json_to_df(ml_experiments_dir, k_list, b_list=b_list)
          else:
              df, avg_strats = classification_report_json_to_df(ml_experiments_dir, k_list, l_list=l_list)
          sample_df_list.append(df)

      for j, col in enumerate(columns):
          multiplier = 0
          non_masked_report_path = output_base_path / 'inputdataset/folds/ml_experiments/experiment_1/classification_report.json'
          with open(non_masked_report_path, 'r') as f:
              non_masked_report = json.load(f)
              y_val = non_masked_report[columns[j]]['mean_f1-score']
              axes[i, j].axhline(y=y_val, label='originele\ndataset', color='red', linestyle='--', zorder=0)

          non_masked_balanced_report_path = output_base_path / 'inputdataset/folds/ml_experiments/experiment_3/balanced/classification_report.json'
          with open(non_masked_balanced_report_path, 'r') as f:
              non_masked_balanced_report = json.load(f)
              y_val = non_masked_balanced_report[columns[j]]['mean_f1-score']
              axes[i, j].axhline(y=y_val, label='gebalanceerde\ndataset', color='black', linestyle='--', zorder=0)

          for k in k_list:
              for r, (df, strat) in enumerate(zip(sample_df_list, strats)):
                  if r == 0:
                      k_rows = df.query(f'k == {k} and b in @b_list').sort_values(by='b', ascending=False)
                  else:
                      l_list = dataset['l_list']
                      k_rows = df.query(f'k == {k} and l in @l_list').sort_values(by='l', ascending=True)
                  mean_list = k_rows[f'{col}.mean_{metric}'].tolist()
                  min_list = k_rows[f'{col}.min_{metric}'].tolist()
                  max_list = k_rows[f'{col}.max_{metric}'].tolist()
                  std_list = k_rows[f'{col}.std_{metric}'].tolist()

                  x = np.arange(len(b_list))

                  offset = width * multiplier
                  lower_errors = [mean - min_val for mean, min_val in zip(mean_list, min_list)]
                  upper_errors = [max_val - mean for mean, max_val in zip(mean_list, max_list)]
                  error_bars = [lower_errors, upper_errors]
                  axes[i, j].bar(x + offset, mean_list, width, label=f'k={k}', yerr=error_bars, capsize=4, color=colors[k], hatch=hatches[r], edgecolor='white')
                  multiplier += 1

              multiplier += k_gap / width

          axes[i, j].set_title(f"{dataset['name']} - {col_names[j]} f1-score")
          axes[i, j].set_xlabel('β\nl', loc='left', labelpad=-21)
          axes[i, j].set_xticklabels(labels)
          x_ticks = x + float(multiplier - 1 - k_gap / width) * width / 2
          axes[i, j].set_xticks(x_ticks)



  legend_elements = [
      Line2D([0], [0], color='red', lw=2, linestyle='--', label='originele\ndataset'),
      Line2D([0], [0], color='black', lw=2, linestyle='--', label='gebalanceerde\ndataset'),
      Patch(facecolor='tab:blue', edgecolor='black', label='k=20'),
      Patch(facecolor='tab:orange', edgecolor='black', label='k=50'),
      Patch(facecolor='tab:green', edgecolor='black', label='k=100'),
      Patch(facecolor='white', hatch=hatches[0], edgecolor='black', label=strat_names[0]),
      Patch(facecolor='white', hatch=hatches[1], edgecolor='black', label=strat_names[1])
  ]

  fig.legend(handles=legend_elements, loc='upper right')
  plt.show()

  
def figure7(include_1=False):
  datasets = [{'name': 'ACSIncome 1664500', 'output_base_path': '../data/output2/ACSIncome_USA_2018_binned_imbalanced_1664500/', 'minority_class': '[100000-inf[', 'minority_class_name': 'rich'},
              {'name': 'Nursery', 'output_base_path': '../data/output2/nursery/', 'minority_class': 'very_recom', 'minority_class_name': 'very_recom'},
              {'name': 'CMC', 'output_base_path': '../data/output2/cmc/', 'minority_class': '2', 'minority_class_name': 'long-term'},
              {'name': 'ACSIncome 16645', 'output_base_path': '../data/output2/ACSIncome_USA_2018_binned_imbalanced_16645/', 'minority_class': '[100000-inf[', 'minority_class_name': 'rich'},
              {'name': 'ACSIncome 1664', 'output_base_path': '../data/output2/ACSIncome_USA_2018_binned_imbalanced_1664/', 'minority_class': '[100000-inf[', 'minority_class_name': 'rich'},
            ]
  
  # sample_strats = ['SSAMPLE_V2', 'BSAMPLE_V2']
  # sample_strat_names = ['ssample_s', 'bsample_s']
  sample_strats = ['BSAMPLE_V2', 'SSAMPLE_V2']
  sample_strat_names = ['bsample_S', 'ssample_S']
  sample_experiment_numbers = [3, 3]
  plot_std = False
  k_list = [10, 20, 50, 100]
  b_list = [0.5, 0.1, 0.01]
  b_list_display = [1] + b_list if include_1 else b_list
  fraction_b_values = [Fraction(value).limit_denominator() for value in b_list_display]
  metric = 'f1-score'
  width = 0.10
  k_gap = 0.015
  # colors = {20: 'tab:blue', 50: 'tab:orange', 100:'tab:green'}
  colors = {10: 'tab:blue', 20: 'tab:orange', 50:'tab:green', 100:'tab:red'}
  hatches = ['\\\\','//']

  fig, axes = plt.subplots(len(datasets), 2, figsize=(15,10), sharey=True, sharex=False)
  plt.subplots_adjust(hspace=0.5, wspace=0.05)

  for i, dataset in enumerate(datasets):
    sample_df_list = []
    columns = ['macro avg', dataset['minority_class']]
    col_names = ['macro avg', dataset['minority_class_name']]

    for t, sample_strat in enumerate(sample_strats):
      output_base_path = Path(dataset['output_base_path']).resolve()
      ml_experiments_dir = output_base_path/f'{sample_strat}/ml_experiments/experiment_{sample_experiment_numbers[t]}'
      df, avg_strats = classification_report_json_to_df(ml_experiments_dir, k_list, b_list)
      if include_1:
        kanon_dir = output_base_path/'kAnon'/'ml_experiments/experiment_1'
        for k in k_list:
          with open(kanon_dir/f'k{k}'/'classification_report.json', 'r') as json_file:
            report = json.load(json_file)
            flat_json = pd.json_normalize(report)
            flat_json['k'] = k
            flat_json['b'] = 1
          df = pd.concat([df, flat_json], ignore_index=True)
      sample_df_list.append(df)

    for j, col in enumerate(columns):
      multiplier = 0
      # non_masked_report_path = output_base_path/'inputdataset/folds/ml_experiments/experiment_1/classification_report.json'
      # with open(non_masked_report_path, 'r') as f:
      #   non_masked_report = json.load(f)
      #   y_val = non_masked_report[columns[j]]['mean_f1-score']
      #   axes[i,j].axhline(y=y_val, label='originele\ndataset', color='red', linestyle='--', zorder=0)
      non_masked_balanced_report_path = output_base_path/'inputdataset/folds/ml_experiments/experiment_3/balanced/classification_report.json'
      with open(non_masked_balanced_report_path, 'r') as f:
        non_masked_balanced_report = json.load(f)
        y_val = non_masked_balanced_report[columns[j]]['mean_f1-score']
        axes[i,j].axhline(y=y_val, label='Originele Dataset\n(Gebalanceerd)', color='red', linestyle='--', zorder=0)

      for k in k_list:   
        for r, (df, sample_strat) in enumerate(zip(sample_df_list, sample_strats)):
          mean_dict, std_dict, min_dict, max_dict = {}, {}, {}, {}
          k_rows = df.query(f'k == {k} and b in @b_list_display').sort_values(by='b', ascending=False)
          mean_dict[k] = k_rows[f'{col}.mean_{metric}'].tolist()
          min_dict[k] = k_rows[f'{col}.min_{metric}'].tolist()
          max_dict[k] = k_rows[f'{col}.max_{metric}'].tolist()
          std_dict[k] = k_rows[f'{col}.std_{metric}'].tolist()
      
          x = np.arange(len(fraction_b_values))
          
          for key, value in mean_dict.items():
            offset = width * multiplier
            min_values = min_dict[key]
            max_values = max_dict[key]
            lower_errors = [mean - min_val for mean, min_val in zip(value, min_values)]
            upper_errors = [max_val - mean for mean, max_val in zip(value, max_values)]
            error_bars = [lower_errors, upper_errors]
            if plot_std:
              # rects = axes[i,j].bar(x + offset, value, width, label=f'k={key}, strat={sample_strat_names[t]}', yerr=std_dict[key])
              rects = axes[i,j].bar(x + offset, value, width, label=f'k={key}', yerr=std_dict[key])
            else:
              # rects = axes[i,j].bar(x + offset, value, width, label=f'k={key}, strat={sample_strat_names[t]}', yerr=error_bars)
              rects = axes[i,j].bar(x + offset, value, width, label=f'k={key}', yerr=error_bars, capsize=4, color=colors[key], hatch=hatches[r], edgecolor='white')
              # for bar in rects:
              #   bar.set_hatch_color('red')
            multiplier += 1
      
        multiplier += k_gap / width

      axes[i,j].set_title(f"{dataset['name']} - {col_names[j]} f1-score")
      axes[i,j].set_xlabel(BETA, loc='left', labelpad=-9)
      axes[i,j].set_xticklabels(fraction_b_values)
      axes[i,j].set_xticks(x + float(multiplier - 1 - k_gap / width) * width / 2)
        
  
  # Adding custom legend entries for hatches
  legend_elements = [
      # Line2D([0], [0], color='red', lw=2, linestyle='--', label='originele\ndataset'),
      Line2D([0], [0], color='red', lw=2, linestyle='--', label='Originele Dataset\n(Gebalanceerd)'),
      Patch(facecolor='tab:blue', edgecolor='black', label='k=10'),
      Patch(facecolor='tab:orange', edgecolor='black', label='k=20'),
      Patch(facecolor='tab:green', edgecolor='black', label='k=50'),
      Patch(facecolor='tab:red', edgecolor='black', label='k=100'),
      Patch(facecolor='white', hatch=hatches[0], edgecolor='black', label=sample_strat_names[0]),
      Patch(facecolor='white', hatch=hatches[1], edgecolor='black', label=sample_strat_names[1])
    ]

  # Create the legend and apply to the plot
  fig.legend(handles=legend_elements, loc='upper right')
  plt.show()

def calculate_xticks(k_list, b_list, num_strats, width, k_gap):
    offset = 0.025
    group_width = num_strats * len(k_list) * width + (len(k_list) - 1) * k_gap
    ticks = []
    # ticks.append(3*width/2 - width/2)
    ticks.append(width+width/2)
    x_pos = 2*width + width/2 + 2*k_gap + 2.5*width +2*offset
    for b in b_list[1:]:
      end_pos = x_pos + group_width
      tick_pos = x_pos + (end_pos - x_pos)/2 - offset
      x_pos = end_pos + width + 3*k_gap/2
      ticks.append(tick_pos)
      offset *= 2
    return ticks

def figure8(include_1=True):
  datasets = [{'name': 'ACSIncome 1664500', 'output_base_path': '../data/output2/ACSIncome_USA_2018_binned_imbalanced_1664500/', 'minority_class': '[100000-inf[', 'minority_class_name': 'rich'},
              {'name': 'ACSIncome 16645', 'output_base_path': '../data/output2/ACSIncome_USA_2018_binned_imbalanced_16645/', 'minority_class': '[100000-inf[', 'minority_class_name': 'rich'},
              {'name': 'ACSIncome 1664', 'output_base_path': '../data/output2/ACSIncome_USA_2018_binned_imbalanced_1664/', 'minority_class': '[100000-inf[', 'minority_class_name': 'rich'},
              {'name': 'Nursery', 'output_base_path': '../data/output2/nursery/', 'minority_class': 'very_recom', 'minority_class_name': 'very_recom'},
              {'name': 'CMC', 'output_base_path': '../data/output2/cmc/', 'minority_class': '2', 'minority_class_name': 'long-term'},
            ]
  
  # sample_strats = ['SSAMPLE_V2', 'BSAMPLE_V2']
  # sample_strat_names = ['ssample_s', 'bsample_s']
  sample_strats = ['BSAMPLE_V2', 'SSAMPLE_V2', ]
  sample_strat_names = ['bsample_s', 'ssample_s']
  sample_experiment_numbers = [3, 3]
  plot_std = False
  k_list = [10, 20, 50, 100]
  b_list = [0.5, 0.1, 0.01]
  b_list_display = [1] + b_list if include_1 else b_list
  fraction_b_values = [Fraction(value).limit_denominator() for value in b_list_display]
  metric = 'f1-score'
  # width = 0.13
  # k_gap = 0.025
  # capsize=4
  width = 0.115
  k_gap = 0.0
  capsize=2.5
  # colors = {20: 'tab:blue', 50: 'tab:orange', 100:'tab:green'}µ
  colors = {10: 'tab:blue', 20: 'tab:orange', 50:'tab:green', 100:'tab:red'}
  hatches = ['\\\\','//']
  
  fig, axes = plt.subplots(len(datasets), 2, figsize=(15,10), sharey=True, sharex=False)
  plt.subplots_adjust(hspace=0.5, wspace=0.05)

  for i, dataset in enumerate(datasets):
    sample_df_list = []
    columns = ['macro avg', dataset['minority_class']]
    col_names = ['macro avg', dataset['minority_class_name']]

    for t, sample_strat in enumerate(sample_strats):
      output_base_path = Path(dataset['output_base_path']).resolve()
      ml_experiments_dir = output_base_path/f'{sample_strat}/ml_experiments/experiment_{sample_experiment_numbers[t]}'
      df, avg_strats = classification_report_json_to_df(ml_experiments_dir, k_list, b_list)
      if include_1:
        kanon_dir = output_base_path/'kAnon'/'ml_experiments/experiment_3'
        for k in k_list:
          with open(kanon_dir/f'k{k}'/'classification_report.json', 'r') as json_file:
            report = json.load(json_file)
            flat_json = pd.json_normalize(report)
            flat_json['k'] = k
            flat_json['b'] = 1
          df = pd.concat([df, flat_json], ignore_index=True)
      sample_df_list.append(df)

    for j, col in enumerate(columns):
      x_ticks = []
      multiplier = 0
      # non_masked_report_path = output_base_path/'inputdataset/folds/ml_experiments/experiment_1/classification_report.json'
      # with open(non_masked_report_path, 'r') as f:
      #   non_masked_report = json.load(f)
      #   y_val = non_masked_report[columns[j]]['mean_f1-score']
      #   axes[i,j].axhline(y=y_val, label='originele\ndataset', color='red', linestyle='--', zorder=0)
      non_masked_balanced_report_path = output_base_path/'inputdataset/folds/ml_experiments/experiment_3/balanced/classification_report.json'
      with open(non_masked_balanced_report_path, 'r') as f:
        non_masked_balanced_report = json.load(f)
        y_val = non_masked_balanced_report[columns[j]]['mean_f1-score']
        axes[i,j].axhline(y=y_val, label='gebalanceerde\ndataset', color='red', linestyle='--', zorder=0)

      # plot b=1
      ssample_df = sample_df_list[0]
      for k in k_list:
        k_rows = ssample_df.query(f'k == {k} and b == 1')
        mean = k_rows[f'{col}.mean_{metric}'].tolist()
        min = k_rows[f'{col}.min_{metric}'].tolist()
        max = k_rows[f'{col}.max_{metric}'].tolist()
        std = k_rows[f'{col}.std_{metric}'].tolist() 
        lower_errors = [mean - min_val for mean, min_val in zip(mean, min)]
        upper_errors = [max_val - mean for mean, max_val in zip(mean, max)]
        error_bars = [lower_errors, upper_errors]
        x = np.arange(1)
        offset = width * multiplier
        rects = axes[i,j].bar(x + offset, mean, width, label=f'k={k}', yerr=error_bars, capsize=capsize, color=colors[k])
        multiplier += 1

      multiplier = 0

      for k in k_list:   
        for r, (df, sample_strat) in enumerate(zip(sample_df_list, sample_strats)):
          k_rows = df.query(f'k == {k} and b in @b_list').sort_values(by='b', ascending=False)
          mean_list = k_rows[f'{col}.mean_{metric}'].tolist()
          min_list = k_rows[f'{col}.min_{metric}'].tolist()
          max_list = k_rows[f'{col}.max_{metric}'].tolist()
          std_list = k_rows[f'{col}.std_{metric}'].tolist() 
        
          x = np.arange(len(fraction_b_values))[1:]
          
          offset = width * multiplier - 3*width - 2*k_gap
          lower_errors = [mean - min_val for mean, min_val in zip(mean_list, min_list)]
          upper_errors = [max_val - mean for mean, max_val in zip(mean_list, max_list)]
          error_bars = [lower_errors, upper_errors]
          if plot_std:
            rects = axes[i,j].bar(x + offset, mean_list, width, label=f'k={k}', yerr=std_list)
          else:
            rects = axes[i,j].bar(x + offset, mean_list, width, label=f'k={k}', yerr=error_bars, capsize=capsize, color=colors[k], hatch=hatches[r], edgecolor='white')
          multiplier += 1

        multiplier += k_gap / width

      axes[i,j].set_title(f"{dataset['name']} - {col_names[j]} F1-score")
      axes[i,j].set_xlabel(BETA, loc='left', labelpad=-9)
      x_ticks = calculate_xticks(k_list, b_list_display, 2, width, k_gap)
      axes[i,j].set_xticks(x_ticks)
      axes[i,j].set_xticklabels(fraction_b_values)
      
      
  # Adding custom legend entries for hatches
  legend_elements = [
      # Line2D([0], [0], color='red', lw=2, linestyle='--', label='original\ndataset'),
      Line2D([0], [0], color='red', lw=2, linestyle='--', label='original\ndataset\n(balanced)'),
      Patch(facecolor='tab:blue', edgecolor='black', label='k=10'),
      Patch(facecolor='tab:orange', edgecolor='black', label='k=20'),
      Patch(facecolor='tab:green', edgecolor='black', label='k=50'),
      Patch(facecolor='tab:red', edgecolor='black', label='k=100'),
      Patch(facecolor='white', hatch=hatches[0], edgecolor='black', label=sample_strat_names[0]),
      Patch(facecolor='white', hatch=hatches[1], edgecolor='black', label=sample_strat_names[1])
    ]

  # Create the legend and apply to the plot
  fig.legend(handles=legend_elements, loc='upper right')
  plt.show()


def figure3():
  datasets = [{'name': 'ACSIncome', 'output_base_path': '../data/output2/ACSIncome_USA_2018_binned_imbalanced_1664500/', 'minority_class': '[100000-inf[', 'minority_class_name': 'rich'},
              {'name': 'Nursery', 'output_base_path': '../data/output2/nursery/', 'minority_class': 'very_recom', 'minority_class_name': 'very_recom'},
              {'name': 'CMC', 'output_base_path': '../data/output2/cmc/', 'minority_class': '2', 'minority_class_name': 'long-term'}]
  
  minority = False
  mutli_col = True
  plot_std = False
  sample_strat = 'BSAMPLE_V2'
  ml_experiment_num = 3
  
  fig, axes = plt.subplots(len(datasets), 2 if mutli_col else 1, figsize=(15,10), sharey=True, sharex=False)
  plt.subplots_adjust(hspace=0.5, wspace=0.05)
  # fig.suptitle(f'macro avg f1-score voor k-anonymity ({BETA}=1) en voor k-anonymity+SSAMPLE_S ({BETA}=0.5)')
  for i, dataset in enumerate(datasets):
    k_list = [10, 20, 50, 100]
    b_list = [0.5, 0.1, 0.01]
    # read_config(config_file)
    output_base_path = Path(dataset['output_base_path']).resolve()
    ml_experiments_dir = output_base_path/f'{sample_strat}/ml_experiments/experiment_{ml_experiment_num}'
    
    df, avg_strats = classification_report_json_to_df(ml_experiments_dir, k_list, b_list)
    kanon_dir = output_base_path/'kAnon'/'ml_experiments'/f'experiment_{ml_experiment_num}'
    b_list.insert(0,1)
    fraction_b_values = [Fraction(value).limit_denominator() for value in b_list]

    for k in k_list:
      with open(kanon_dir/f'k{k}'/'classification_report.json', 'r') as json_file:
        report = json.load(json_file)
        flat_json = pd.json_normalize(report)
        flat_json['k'] = k
        flat_json['b'] = 1
      df = pd.concat([df, flat_json], ignore_index=True)
    
    metric = 'f1-score'
    columns = ['macro avg', dataset['minority_class']] if mutli_col else ['macro avg']
    col_names = ['macro avg', dataset['minority_class_name']]
    
    for j, col in enumerate(columns):
      mean_dict, std_dict, min_dict, max_dict = {}, {}, {}, {}

      for k in k_list:
        k_rows = df.query(f'k == {k} and b in @b_list').sort_values(by='b', ascending=False)
        if minority:
          mean_dict[k] = k_rows[f'{dataset["minority_class"]}.mean_{metric}'].tolist()
          min_dict[k] = k_rows[f'{dataset["minority_class"]}.min_{metric}'].tolist()
          max_dict[k] = k_rows[f'{dataset["minority_class"]}.max_{metric}'].tolist()
        elif mutli_col:
          mean_dict[k] = k_rows[f'{col}.mean_{metric}'].tolist()
          min_dict[k] = k_rows[f'{col}.min_{metric}'].tolist()
          max_dict[k] = k_rows[f'{col}.max_{metric}'].tolist()
          std_dict[k] = k_rows[f'{col}.std_{metric}'].tolist()
        else:
          mean_dict[k] = k_rows[f'macro avg.mean_{metric}'].tolist()
          min_dict[k] = k_rows[f'macro avg.min_{metric}'].tolist()
          max_dict[k] = k_rows[f'macro avg.max_{metric}'].tolist()
      
      x = np.arange(len(fraction_b_values))
      
      width = 0.13
      multiplier = 0

      for key, value in mean_dict.items():
        offset = width * multiplier
        min_values = min_dict[key]
        max_values = max_dict[key]
        lower_errors = [mean - min_val for mean, min_val in zip(value, min_values)]
        upper_errors = [max_val - mean for mean, max_val in zip(value, max_values)]
        error_bars = [lower_errors, upper_errors]
        if mutli_col:
          if plot_std:
            rects = axes[i,j].bar(x + offset, value, width, label=f'k={key}', yerr=std_dict[key])
          else:
            rects = axes[i,j].bar(x + offset, value, width, label=f'k={key}', yerr=error_bars, capsize=4)
        else:
          rects = axes[i].bar(x + offset, value, width, label=f'k={key}', yerr=error_bars, capsize=4)
        multiplier += 1

      # non_masked_report_path = output_base_path/'inputdataset/folds/ml_experiments/experiment_1/classification_report.json'
      # with open(non_masked_report_path, 'r') as f:
      #   non_masked_report = json.load(f)
      #   if mutli_col:
      #     y_val = non_masked_report[col]['mean_f1-score']
      #     axes[i,j].axhline(y=y_val, label='originele\n dataset', color='red', linestyle='--')
      #   elif minority:
      #     y_val = non_masked_report[f'{dataset["minority_class"]}']['mean_f1-score']
      #     axes[i].axhline(y=y_val, label='originele\n dataset', color='red', linestyle='--')
      #   else:
      #     y_val = non_masked_report['macro avg']['mean_f1-score']
      #     axes[i].axhline(y=y_val, label='originele\n dataset', color='red', linestyle='--')
      
      non_masked_report_path_balanced = output_base_path/'inputdataset/folds/ml_experiments/experiment_3/balanced/classification_report.json'
      with open(non_masked_report_path_balanced, 'r') as f:
        non_masked_report = json.load(f)
        if mutli_col:
          y_val = non_masked_report[col]['mean_f1-score']
          axes[i,j].axhline(y=y_val, label='Originele Dataset\n(Gebalanceerd)', color='red', linestyle='--')

      if mutli_col:
        axes[i,j].set_title(f"{dataset['name']} - {col_names[j]} f1-score")
        axes[i,j].set_xlabel(BETA, loc='left', labelpad=-9)
        axes[i,j].set_xticklabels(fraction_b_values)
        axes[i,j].set_xticks(ticks=x+float((multiplier-1)/2)*width)
      else:
        axes[i].set_title(dataset['name'])
        axes[i].set_xlabel(BETA, loc='left', labelpad=-9)
        axes[i].set_xticklabels(fraction_b_values)
        axes[i].set_xticks(ticks=x+float((multiplier-1)/2)*width)
  
  if mutli_col:
    handles, labels = axes[0,0].get_legend_handles_labels()
  else:
    handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc='upper right')
  plt.show()


def draw_eq_distributions(k_list, k_anon_base_path, output_base_path, qid_list, log_scale=False):
	for k in k_list:
		kanon_sample_path = k_anon_base_path/'fold_0'/f'k{k}'/'output_sample.csv'
		output_path = output_base_path/'stats'/'kAnon'/'EQdist'/f'{k}_eq_distribution.png'
		output_path.parent.mkdir(parents=True, exist_ok=True)
		kanon_sample_df = pd.read_csv(kanon_sample_path, sep=';', decimal=',')
		eq_sizes = kanon_sample_df.groupby(qid_list).size()

		plt.figure(figsize=(10, 6))
		min_eq_size = eq_sizes.min()  # Smallest EQ size
		max_eq_size = eq_sizes.max()  # Largest EQ size
		bins = np.arange(min_eq_size - 0.5, max_eq_size + 1.5)
		plt.hist(eq_sizes, bins=bins, alpha=0.75, color='blue', edgecolor='black')
		
		plt.title(f'EQ-grootte Distributie voor k={k}')
		plt.xlabel('EQ-grootte')
		plt.ylabel('Frequentie')
		plt.xlim(min_eq_size - 1, max_eq_size + 1)	
		if log_scale:
			plt.xscale('log')
		# plt.xlim(min_eq_size - 1, 37)
		plt.savefig(output_path)



def get_qid(experiment_stats_path) -> list:
  with open(experiment_stats_path, 'r') as json_file:
    stats = json.load(json_file)
  return stats.get('qid', [])



def eq_distribution():
  k = 50
  fold = 0
  datasets = [
        {'name': 'ACSIncome', 'output_base_path': '../data/output2/ACSIncome_USA_2018_binned_imbalanced_1664500/', 'minority_class': '[100000-inf[', 'minority_class_name': 'rich', 'color': 'tab:blue'},
        {'name': 'Nursery', 'output_base_path': '../data/output2/nursery/', 'minority_class': 'very_recom', 'minority_class_name': 'very_recom', 'color': 'tab:orange'},
        {'name': 'CMC', 'output_base_path': '../data/output2/cmc/', 'minority_class': '2', 'minority_class_name': 'long-term', 'color': 'purple'}
    ]
  fig, ax = plt.subplots(figsize=(15,10))

  max_eq_size = 0

  for i, dataset in enumerate(datasets):
    output_base_path = Path(dataset['output_base_path']).resolve()
    kanon_sample_path = output_base_path/'kAnon'/'fold_0'/f'k{k}'/'output_sample.csv'
    experiment_stats_path = output_base_path/'stats.json'
    kanon_sample_df = pd.read_csv(kanon_sample_path, sep=';', decimal=',')
    eq_sizes = kanon_sample_df.groupby(get_qid(experiment_stats_path)).size()
    max_eq_size = max(max_eq_size, eq_sizes.max())
    max_eq_size = eq_sizes.max()
    datasets[i]['eq_sizes'] = eq_sizes

  bins = np.arange(0, max_eq_size + 1)
  for i, dataset in enumerate(datasets):
     ax.hist(dataset['eq_sizes'], bins=bins, label=dataset['name'], edgecolor='black', color=dataset['color'])
  
  ax.set_xlabel('EQ Size', fontsize=14)  
  ax.set_ylabel('Count', fontsize=14) 
  ax.legend(fontsize=12)
  ax.tick_params(axis='both', which='major', labelsize=12)  
  ax.set_yscale('log')
  ax.set_xlim(k, 220)

  current_ticks = list(ax.get_xticks())
  if k not in current_ticks:
      current_ticks.append(k)  
      current_ticks.sort() 
  ax.set_xticks(current_ticks)  

  plt.show()




if __name__ == '__main__':
  # eq_distribution()
  # figure3()
  # figure7()
  # figure8()
  ldiv_compare_bsample_2()