import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
import shutil
import json
from pprint import pprint
from multiprocessing import Pool
from tqdm import tqdm

from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from joblib import dump
import pickle
from copy import deepcopy

from configparser import ConfigParser, ExtendedInterpolation

from stats import generalization_stats, sample_stats, eq_per_target


def read_config(config_path):
  global CONFIG_PATH
  CONFIG_PATH = config_path
  cfg = ConfigParser(interpolation=ExtendedInterpolation())
  cfg.read(CONFIG_PATH)
  
  # PATHS
  global OUTPUT_BASE_PATH, EXPERIMENT_STATS_PATH, HIERARCHIES_BASE_PATH, TRAIN_DF_PATH, INPUT_DATA_DEFENITION_PATH, TEST_DF_PATH, K_ANON_BASE_PATH, FOLDS_PATH
  OUTPUT_BASE_PATH = Path(cfg['PATHS']['output_base_path']).resolve()
  K_ANON_BASE_PATH = OUTPUT_BASE_PATH / 'kAnon'
  EXPERIMENT_STATS_PATH = Path(cfg['PATHS']['experiment_stats_path']).resolve()
  HIERARCHIES_BASE_PATH = Path(cfg['PATHS']['hierarchies_base_path']).resolve()
  FOLDS_PATH = Path(cfg['PATHS']['folds_path']).resolve()

  # BOOLEANS
  global MLBALANCE, PRIVACY_METRICS, ML, SAVE_TEST_GENERALIZED, APPEND_ML_EXPERIMENTS
  MLBALANCE = cfg.getboolean('BOOLEANS', 'mlbalance')
  PRIVACY_METRICS = cfg.getboolean('BOOLEANS', 'privacy_metrics')
  ML = cfg.getboolean('BOOLEANS', 'ml')
  SAVE_TEST_GENERALIZED = cfg.getboolean('BOOLEANS', 'save_test_generalized')
  APPEND_ML_EXPERIMENTS = cfg.getboolean('BOOLEANS', 'append_ml_experiments')

  # VARIABLES
  global NUM_PROCESSES, NUM_FOLDS, DATASET_NAME, K_LIST, B_LIST, L_LIST, TARGETS, EXP_NUMBERS
  # NUM_PROCESSES = cfg.getint('VARIABLES', 'num_processes')
  NUM_FOLDS = cfg.getint('VARIABLES', 'num_folds')
  DATASET_NAME= cfg['VARIABLES']['dataset_name']
  
  K_LIST = [int(num) for num in cfg.get('VARIABLES', 'k_list').split(',')]
  B_LIST = [float(num) for num in cfg.get('VARIABLES', 'b_list').split(',')]
  L_LIST = [float(num) for num in cfg.get('VARIABLES', 'l_list').split(',')]

  # K_LIST = [5, 10, 20, 40, 80, 160]
  # B_LIST = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
  # L_LIST = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
  TARGETS = cfg.get('VARIABLES', 'targets').split(',')
  if DATASET_NAME == 'cmc':
    TARGETS = [int(target) for target in TARGETS]
  EXP_NUMBERS = [int(num) for num in cfg.get('VARIABLES', 'experiment_numbers').split(',')]

  # OTHER
  global QID_LIST, CAT, NUM, TARGET
  QID_LIST = get_qid()
  CAT, NUM, TARGET = get_datatypes()
  
def get_pipelines(experiment_num, verbose=False):
  if(DATASET_NAME=='cmc'):
    from pipelines.CMC_pipelines import experiment_1_pipeline, experiment_2_pipeline, experiment_3_pipeline
    if experiment_num == 1:
      return experiment_1_pipeline(verbose, QID_LIST)
    elif experiment_num == 2:
      return experiment_2_pipeline(verbose, QID_LIST)
    elif experiment_num == 3:
      return experiment_3_pipeline(verbose, QID_LIST)
  
  elif(DATASET_NAME=='ASCIncome'):
    from pipelines.ASCIncome_pipelines import experiment_1_pipeline, experiment_3_pipeline
    if experiment_num == 1:
      return experiment_1_pipeline(verbose, QID_LIST, CAT)
    elif experiment_num == 2:
      return experiment_2_pipeline(verbose, QID_LIST, CAT)
    elif experiment_num == 3:
      return experiment_3_pipeline(verbose, QID_LIST, CAT)
    
  elif(DATASET_NAME=='nursery'):
    from pipelines.nursery_pipelines import experiment_1_pipeline, experiment_3_pipeline
    if experiment_num == 1:
      return experiment_1_pipeline(verbose, QID_LIST, CAT)
    elif experiment_num == 2:
      return experiment_2_pipeline(verbose, QID_LIST, CAT)
    elif experiment_num == 3:
      return experiment_3_pipeline(verbose, QID_LIST, CAT)

def get_xy_split(df_to_split: pd.DataFrame, target):
  x, y = df_to_split.drop(target, axis=1), df_to_split[target]
  return (x,y)
  
def get_datatypes():
  experiment_stats = pd.read_csv(EXPERIMENT_STATS_PATH, delimiter=';')
  global INPUT_DATA_DEFENITION_PATH
  INPUT_DATA_DEFENITION_PATH = experiment_stats['inputDataDefenitionAbsolutePath'].to_list()[0]
  with open(INPUT_DATA_DEFENITION_PATH, 'r') as json_file:
    data_defenition = json.load(json_file)
  cat = data_defenition.get('cat', [])
  num = data_defenition.get('num', [])
  target = data_defenition.get('target')
  return (cat, num, target)
  

def certainty(population_df: pd.DataFrame, sample_df: pd.DataFrame, attributes: list) -> pd.Series:
  population_eq = population_df.groupby(attributes).size()
  sample_eq = sample_df.groupby(attributes).size()

  result = sample_eq.div(population_eq).fillna(0).reset_index(level=attributes)
  result2 = sample_df.merge(result, on=attributes, how='left')
  return result2[0]


def journalist_risk(population_df: pd.DataFrame, sample_df: pd.DataFrame, attributes: list) -> pd.Series:
  population_eq = population_df.groupby(attributes).size().apply(lambda x: 1 / x).reset_index(level=attributes)

  result = sample_df.merge(population_eq, on=attributes, how='left').fillna(0)
  suppressed_records = sample_df[sample_df[attributes] == '*'][attributes].dropna()
  result.iloc[suppressed_records.index, -1] = 0
  return result[result.columns[-1]]

def read_attributes(settingsPath: str, attributeType: str) -> list:
  settings = pd.read_csv(settingsPath, delimiter=';')
  attributes = settings[attributeType].to_list()[0][1:-1].split(', ')
  return attributes

def get_qid() -> list:
  experiment_stats = pd.read_csv(EXPERIMENT_STATS_PATH, delimiter=';')
  return experiment_stats["QID"].to_list()[0][1:-1].split(', ')

def get_generalised(dataset: pd.DataFrame, stats_file: Path, qid_list, hierarchies_base_path) -> pd.DataFrame:
  dataset = dataset.copy()
  stats_df = pd.read_csv(stats_file, sep=';', decimal=',')
  gen_levels = stats_df['node'].tolist()[0][1:-1].split(', ')
  gen_levels = [int(level) for level in gen_levels]
  
  for level, qi in zip(gen_levels, qid_list):
    hierarchy = pd.read_csv(hierarchies_base_path/f'{qi}.csv', sep=';', decimal=',', header=None, dtype=str) #pay attention to dtype
    hierarchy.set_index(hierarchy.columns[0], drop=False, inplace=True)
    generalization = hierarchy[level]
    dataset[qi] = dataset[qi].map(generalization)
  return dataset

def get_stats_file(k_dir_name: str, fold_num=None):
  if fold_num==None:
    return K_ANON_BASE_PATH / k_dir_name / 'stats.csv'
  else:
    return K_ANON_BASE_PATH / f'fold_{fold_num}' / k_dir_name / 'stats.csv'

def calculate_privacy_metrics(sample_strategy):
  sample_base_path = OUTPUT_BASE_PATH / sample_strategy

  samples = []
  k_directories = [sample_base_path / d for d in os.listdir(sample_base_path)]

  for k_dir in k_directories:
    k_stats_file_path = OUTPUT_BASE_PATH / 'kAnon' / k_dir.name / 'stats.csv'
    dict = {'stats_file_path': k_stats_file_path, "sample_paths": []}
    for sample_folder_name in os.listdir(k_dir):
      dict['sample_paths'].append(Path(k_dir) / sample_folder_name / f'{sample_folder_name}_sample.csv')
    samples.append(dict)

  for dict in samples:
    population_df_generalized = get_generalised(POPULATION_DF, dict['stats_file_path'])
    for sample_path in dict['sample_paths']:
      privacystats_dir = sample_path.parent / 'privacystats'
      if privacystats_dir.exists():
        shutil.rmtree(privacystats_dir)
      os.makedirs(privacystats_dir)
      
      sample_df = pd.read_csv(sample_path, delimiter=';')
      certainty_distribution = certainty(population_df_generalized, sample_df, QID_LIST)
      certainty_distribution.to_csv(privacystats_dir / 'certainty.csv', sep=';', decimal=',', index=False)
      
      journalist_risk_distribution = journalist_risk(population_df_generalized, sample_df, QID_LIST)
      journalist_risk_distribution.to_csv(privacystats_dir / 'journalistRisk.csv', sep=';', decimal=',', index=False)
  

def save_ml_experiment(experiment_dir: Path, pipe: Pipeline, report):
  if experiment_dir.exists():
    shutil.rmtree(experiment_dir)
  os.makedirs(experiment_dir)	

  dump(pipe.named_steps['model'], experiment_dir / 'model.joblib')
  with open (experiment_dir/'pipeline.pkl', 'wb') as f:
    pickle.dump(pipe, f)

  with open(experiment_dir/'classification_report.json', 'w') as json_file:
    json.dump(report, json_file, indent=2)

  with open(experiment_dir/'pipeline.txt', 'w') as f:
    pprint(pipe.get_params(), stream=f)


def calculate_mean_std(reports, output_path, targets):
  targets = deepcopy(targets)
  os.makedirs(output_path.parent, exist_ok=True)
  avg_report = {}
  targets += ['macro avg', 'weighted avg']
  for target in targets:
      avg_report[target] = {
          'mean_precision': np.average([report[target]['precision'] for report in reports]),
          'std_precision': np.std([report[target]['precision'] for report in reports]),
          'mean_recall': np.average([report[target]['recall'] for report in reports]),
          'std_recall': np.std([report[target]['recall'] for report in reports]),
          'mean_f1-score': np.average([report[target]['f1-score'] for report in reports]),
          'std_f1-score': np.std([report[target]['f1-score'] for report in reports])
      }

  with open(output_path, 'w') as json_file:
    json.dump(avg_report, json_file, indent=2)	

def get_ml_run_report(y_test_generalized, y_predict, y_train):
  y_train = y_train.astype(str)
  report = classification_report(y_test_generalized, y_predict, output_dict=True)
  return report

def run_model_cv(params):
  experiment_num, sample_path, stats_file_path, test_df_path, target, pipe, qid_list, hierarchies_base_path = params
  train_df = pd.read_csv(sample_path, sep=';')
  test_df = pd.read_csv(test_df_path, sep=';', dtype=str)
  test_df_generalized = get_generalised(test_df, stats_file_path, qid_list, hierarchies_base_path)
  X_train, y_train = get_xy_split(train_df, target)
  X_test_generalized, y_test_generalized = get_xy_split(test_df_generalized, target)

  pipe.fit(X_train, y_train)
  y_predict = pipe.predict(X_test_generalized)
  y_predict = y_predict.astype(str)

  report = get_ml_run_report(y_test_generalized, y_predict, y_train)
  save_ml_experiment(sample_path.parent/'ml_experiments'/f'experiment_{experiment_num}', pipe, report)
  return report

def run_model_cv_non_generalized(params):
  experiment_num, fold_path, target, pipe = params
  train_df = pd.read_csv(fold_path/'train.csv', sep=';')
  test_df = pd.read_csv(fold_path/'test.csv', sep=';')
  X_train, y_train = get_xy_split(train_df, target)
  X_test, y_test = get_xy_split(test_df, target)
  pipe.fit(X_train, y_train)
  y_predict = pipe.predict(X_test)
  report = get_ml_run_report(y_test, y_predict, y_train)
  save_ml_experiment(fold_path/'ml_experiments'/f'experiment_{experiment_num}', pipe, report)
  return report

def parallel_run_model_cv(jobs, non_generalized, progressbar_desc=''):
  target_func = run_model_cv_non_generalized if non_generalized else run_model_cv
  with Pool(processes=NUM_PROCESSES) as pool:
	  return list(tqdm(pool.imap(target_func, jobs),total=len(jobs),desc=progressbar_desc))

def ml_worker_cv_nonmasked(experiment_num, verbose=False):
  pipes_tuple = get_pipelines(experiment_num)
  if experiment_num==3:
    pipe, non_masked_pipe, non_masked_balanced_ROS_pipe = pipes_tuple
    pipes = [non_masked_pipe, non_masked_balanced_ROS_pipe]
    output_paths = [
      FOLDS_PATH/'ml_experiments'/f'experiment_{experiment_num}'/'non_balanced'/'classification_report.json',
      FOLDS_PATH/'ml_experiments'/f'experiment_{experiment_num}'/'balanced'/'classification_report.json'
      ]
    for i in range(2):
      jobs = []
      for fold in range(NUM_FOLDS):
        fold_path = FOLDS_PATH/f'fold_{fold}'
        jobs.append((experiment_num, fold_path, TARGET, deepcopy(pipes[i])))
      reports = parallel_run_model_cv(jobs, non_generalized=True, progressbar_desc=f'non_generalized: True')
      calculate_mean_std(reports, output_paths[i], TARGETS)
  else:
    pipe, non_masked_pipe = pipes_tuple
    output_path = FOLDS_PATH/'ml_experiments'/f'experiment_{experiment_num}'/'classification_report.json'
    jobs = []
    for fold in range(NUM_FOLDS):
      fold_path = FOLDS_PATH/f'fold_{fold}'
      jobs.append((experiment_num, fold_path, TARGET, deepcopy(non_masked_pipe)))
    reports = parallel_run_model_cv(jobs, non_generalized=True, progressbar_desc=f'non_generalized: True')
    calculate_mean_std(reports, output_path, TARGETS)

def ml_worker_cv(experiment_num, strats_to_run, verbose=False):
  pipe, *_ = get_pipelines(experiment_num)
  for strat in strats_to_run:
    for k in K_LIST:
      for b in B_LIST:
        jobs = []
        avg_classification_report_output_path = OUTPUT_BASE_PATH/strat/'ml_experiments'/f'experiment_{experiment_num}'/f'k{k}'/f'B({b})'/'classification_report.json'
        for fold in range(NUM_FOLDS):
          sample_path = OUTPUT_BASE_PATH/strat/f'fold_{fold}'/f'k{k}'/f'B({b})'/f'B({b})_sample.csv'
          stats_file_path = get_stats_file(f'k{k}', fold)
          test_df_path = FOLDS_PATH/f'fold_{fold}'/'test.csv'
          jobs.append((experiment_num, sample_path, stats_file_path, test_df_path, TARGET, deepcopy(pipe), QID_LIST, HIERARCHIES_BASE_PATH))
        reports = parallel_run_model_cv(jobs, non_generalized=False, progressbar_desc=f'strat: {strat}, k: {k}, b: {b}')
        calculate_mean_std(reports, avg_classification_report_output_path, TARGETS)


def ml_worker_cv_ldiv(experiment_num, verbose=False):
  pipe, *_ = get_pipelines(experiment_num)
  for k in K_LIST:
    for l in L_LIST:
      jobs = []
      avg_classification_report_output_path = OUTPUT_BASE_PATH/'lDiv'/'ml_experiments'/f'experiment_{experiment_num}'/f'k{k}'/f'l{l}'/'classification_report.json'
      for fold in range(NUM_FOLDS):
        sample_path = OUTPUT_BASE_PATH/'lDiv'/f'fold_{fold}'/f'k{k}'/f'l{l}'/'sample.csv'
        stats_file_path = OUTPUT_BASE_PATH/'lDiv'/f'fold_{fold}'/f'k{k}'/f'l{l}'/'stats.csv'
        test_df_path = FOLDS_PATH/f'fold_{fold}'/'test.csv'
        jobs.append((experiment_num, sample_path, stats_file_path, test_df_path, TARGET, deepcopy(pipe), QID_LIST, HIERARCHIES_BASE_PATH))
      reports = parallel_run_model_cv(jobs, non_generalized=False, progressbar_desc=f'strat: l-diversity, k: {k}, l: {l} ')
      calculate_mean_std(reports, avg_classification_report_output_path, TARGETS)


if __name__ == '__main__':
  config_path = sys.argv[1]
  NUM_PROCESSES = int(sys.argv[2])
  
  # config_path = 'config/nursery.ini'
  # config_path = 'config\ACSIncome_USA_2018_binned_imbalanced_16645_acc_metric.ini'
  # config_path = 'config/cmc.ini'
  # config_path = 'config\ACSIncome_USA_2018_binned_imbalanced_16645.ini'
  # config_path = 'config/income_binned_USA_1664500.ini'
  read_config(config_path)

  #generalization_stats(K_LIST, NUM_FOLDS, K_ANON_BASE_PATH, OUTPUT_BASE_PATH, QID_LIST)
  #sample_stats(['SSAMPLE', 'BSample'], K_LIST, B_LIST, NUM_FOLDS, OUTPUT_BASE_PATH, QID_LIST, TARGET, TARGETS)
  #eq_per_target(['SSAMPLE', 'BSample'], K_LIST, B_LIST, NUM_FOLDS, OUTPUT_BASE_PATH, QID_LIST, TARGET, TARGETS)

  if PRIVACY_METRICS:
      calculate_privacy_metrics('SSAMPLE', 'BSample')

  for exp_number in EXP_NUMBERS:
    # zou ik hier ook nog een pool kunnen maken? 
    if ML:
      ml_worker_cv_nonmasked(exp_number)
      ml_worker_cv(exp_number, ['SSAMPLE', 'BSample'])
      ml_worker_cv_ldiv(exp_number)
    
  
  
  
