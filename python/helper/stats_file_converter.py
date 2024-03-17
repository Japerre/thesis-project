import csv
import json
from pathlib import Path
import os


kAnon_base_path = Path(f'../../data/output/ACSIncome_USA_2018_binned_imbalanced_1664500/kAnon').resolve()

stats_files_csv = []
stats_files_json = []

for folder, subfolders, files in os.walk(kAnon_base_path):
    for subfolder in subfolders:
        if subfolder.startswith('k'):
            stats_files_csv.append(Path(folder)/subfolder/'stats.csv')
            stats_files_json.append(Path(folder)/subfolder/'stats.json')
            

for stats_file_csv, stats_file_json in zip(stats_files_csv, stats_files_json):
  with open(stats_file_csv, mode='r') as csv_file:
      csv_reader = csv.DictReader(csv_file, delimiter=';')
      data_list = []
      
      for row in csv_reader:
          node_list = [int(x) for x in row['node'].strip('[]').split(', ')]
          qid_list = row['QID'].strip('[]').split(', ')
          
          json_dict = {
              "node": node_list,
              "QID": qid_list,
              "suppressed in sample": int(row['suppressed in sample']),
              "sample size": int(row['sample size']),
              "input size": int(row['input size']),
              "equivalence classes": int(row['equivalence classes']),
              "average EQ size": float(row['average EQ size'])
          }
          
          data_list.append(json_dict)

  if data_list:
      with open(stats_file_json, mode='w') as json_file:
          json.dump(data_list[0], json_file, indent=2)
