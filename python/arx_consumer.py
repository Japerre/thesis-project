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

from stats import generalization_stats, sample_stats, eq_per_target, find_biggest_certainty, find_biggest_procentual_certainty, print_distributions_worker, print_fully_suppressed_samples
from stats import eq_stats, draw_eq_distributions, draw_combined_eq_distributions
from sampleV2 import SampleV2

NUM_PROCESSES = None

def read_config(config_path):
  global CONFIG_PATH, CFG
  CONFIG_PATH = config_path
  CFG = ConfigParser(interpolation=ExtendedInterpolation())
  CFG.read(CONFIG_PATH)
  
  # PATHS
  global OUTPUT_BASE_PATH, EXPERIMENT_STATS_PATH, HIERARCHIES_BASE_PATH, INPUT_DATA_DEFINITION_PATH, K_ANON_BASE_PATH, FOLDS_PATH
  OUTPUT_BASE_PATH = Path(CFG['PATHS']['output_base_path']).resolve()
  K_ANON_BASE_PATH = OUTPUT_BASE_PATH / 'kAnon'
  EXPERIMENT_STATS_PATH = Path(CFG['PATHS']['experiment_stats_path']).resolve()
  INPUT_DATA_DEFINITION_PATH = Path(CFG['PATHS']['data_definition_path']).resolve()
  HIERARCHIES_BASE_PATH = Path(CFG['PATHS']['hierarchies_base_path']).resolve()
  FOLDS_PATH = Path(CFG['PATHS']['folds_path']).resolve()

  # BOOLEANS
  global MLBALANCE, PRIVACY_METRICS, ML, SAVE_TEST_GENERALIZED, APPEND_ML_EXPERIMENTS, RUNSAMPLEV2
  MLBALANCE = CFG.getboolean('BOOLEANS', 'mlbalance')
  PRIVACY_METRICS = CFG.getboolean('BOOLEANS', 'privacy_metrics')
  ML = CFG.getboolean('BOOLEANS', 'ml')
  SAVE_TEST_GENERALIZED = CFG.getboolean('BOOLEANS', 'save_test_generalized')
  APPEND_ML_EXPERIMENTS = CFG.getboolean('BOOLEANS', 'append_ml_experiments')
  RUNSAMPLEV2 = CFG.getboolean('BOOLEANS', 'run_sample_v2')

  # VARIABLES
  global NUM_FOLDS, DATASET_NAME, K_LIST, B_LIST, L_LIST, TARGETS, EXP_NUMBERS
  NUM_FOLDS = CFG.getint('VARIABLES', 'num_folds')
  DATASET_NAME= CFG['PATHS']['dataset_name']
  K_LIST, B_LIST, L_LIST = get_run_params()

  TARGETS = CFG.get('VARIABLES', 'targets').split(',')
  if DATASET_NAME == 'cmc':
    TARGETS = [str(target) for target in TARGETS]
  EXP_NUMBERS = [int(num) for num in CFG.get('VARIABLES', 'experiment_numbers').split(',')]

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
  
  elif(DATASET_NAME=='ACSIncome'):
    from pipelines.ACSIncome_pipelines import experiment_1_pipeline, experiment_3_pipeline
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
  
def get_run_params():
  with open(EXPERIMENT_STATS_PATH, 'r') as json_file:
    stats = json.load(json_file)
  return (stats.get('k_values'), stats.get('b_values'), stats.get('l_values'))

def get_qid() -> list:
  with open(EXPERIMENT_STATS_PATH, 'r') as json_file:
    stats = json.load(json_file)
  return stats.get('qid', [])
  
def get_datatypes():
  with open(INPUT_DATA_DEFINITION_PATH, 'r') as json_file:
    data_definition = json.load(json_file)
  cat = data_definition.get('cat', [])
  num = data_definition.get('num', [])
  target = data_definition.get('target')
  return (cat, num, target)
  
def calculate_privacy_metrics(params):
  population_path, sample_path, journalist_risk, certainty, privacystats_dir, qid_list = params
  population_df = pd.read_csv(population_path, sep=';', decimal=',').astype(str)
  sample_df = pd.read_csv(sample_path, delimiter=';').astype(str)
  # sample_df = pd.read_csv(sample_path, delimiter=';') moest dit veraderen voor nursery

  if certainty:
    certainty_distribution = calc_certainty(population_df, sample_df, qid_list)
    certainty_distribution.to_csv(privacystats_dir / 'certainty.csv', sep=';', decimal=',', index=False)

  if journalist_risk:
    journalist_risk_distribution = calc_journalist_risk(population_df, sample_df, qid_list)
    journalist_risk_distribution.to_csv(privacystats_dir / 'journalistRisk.csv', sep=';', decimal=',', index=False)


def calculate_privacy_metrics_worker(sample_strategies, journalist_risk: bool, certainty: bool):
  jobs = []
  for sample_strat in sample_strategies:
    for fold in range(NUM_FOLDS):
      fold_path = OUTPUT_BASE_PATH/sample_strat/f'fold_{fold}'
      samples = []
      k_directories = [fold_path/d for d in os.listdir(fold_path) if d.startswith('k')]
      for k_dir in k_directories:
        population_path = K_ANON_BASE_PATH/f'fold_{fold}/{k_dir.name}/output_sample.csv'
        dict = {'population_path': population_path, "sample_paths": []}
        for sample_folder_name in os.listdir(k_dir):
          dict['sample_paths'].append(Path(k_dir) / sample_folder_name / f'{sample_folder_name}_sample.csv')
        samples.append(dict)

      for dict in samples:
        for sample_path in dict['sample_paths']:
          privacystats_dir = sample_path.parent / 'privacystats'
          if privacystats_dir.exists():
            shutil.rmtree(privacystats_dir)
          os.makedirs(privacystats_dir)
          jobs.append((dict['population_path'], sample_path, journalist_risk, certainty, privacystats_dir, QID_LIST))

  with Pool(processes=NUM_PROCESSES) as pool:
	  return list(tqdm(pool.imap(calculate_privacy_metrics, jobs),total=len(jobs),desc='calculating privacy metrics'))
  
  

def calc_certainty(population_df: pd.DataFrame, sample_df: pd.DataFrame, qid: list) -> pd.Series:
  population_eq = population_df.groupby(qid).size()
  sample_eq = sample_df.groupby(qid).size()

  result = sample_eq.div(population_eq).fillna(0).reset_index(level=qid)
  result2 = sample_df.merge(result, on=qid, how='left')
  return result2[0]


def calc_journalist_risk(population_df: pd.DataFrame, sample_df: pd.DataFrame, attributes: list) -> pd.Series:
  population_eq = population_df.groupby(attributes).size().apply(lambda x: 1 / x).reset_index(level=attributes)

  result = sample_df.merge(population_eq, on=attributes, how='left').fillna(0)
  suppressed_records = sample_df[sample_df[attributes] == '*'][attributes].dropna()
  result.iloc[suppressed_records.index, -1] = 0
  return result[result.columns[-1]]


def get_generalised(dataset: pd.DataFrame, stats_file: Path, hierarchies_base_path) -> pd.DataFrame:
  dataset = dataset.copy()
  with open(stats_file, 'r') as json_file:
    stats = json.load(json_file)
  gen_levels = stats.get('node', [])
  qid_list = stats.get('QID', [])

  for level, qi in zip(gen_levels, qid_list):
    hierarchy = pd.read_csv(hierarchies_base_path/f'{qi}.csv', sep=';', decimal=',', header=None, dtype=str) #pay attention to dtype
    hierarchy.set_index(hierarchy.columns[0], drop=False, inplace=True)
    generalization = hierarchy[level]
    dataset[qi] = dataset[qi].map(generalization)
  return dataset


def sample_size_zero(stats_file_path=None, sample_path=None, target=None):
  if stats_file_path:
    with open(stats_file_path, 'r') as json_file:
      stats = json.load(json_file)
    return stats['sample size'] == 0
  elif sample_path:
    sample_df = pd.read_csv(sample_path, sep=';', decimal=',')
    if sample_df.empty:
      return True
    elif sample_df[target].nunique() < 2:
      return True
    

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


def calculate_mean_std(reports, output_path, targets, num_folds):
  targets = deepcopy(targets)
  os.makedirs(output_path.parent, exist_ok=True)
  avg_report = {}
  targets += ['macro avg', 'weighted avg']

  for target in targets:
    precisions = [report[target]['precision'] for report in reports]
    recalls = [report[target]['recall'] for report in reports]
    f1_scores = [report[target]['f1-score'] for report in reports]

    # Pad the data with zeros if fewer than desired samples
    non_empty_folds_len = len(precisions)
    precisions += [0] * (num_folds - non_empty_folds_len)
    recalls += [0] * (num_folds - non_empty_folds_len)
    f1_scores += [0] * (num_folds - non_empty_folds_len)
    
    # Calculate mean and standard deviation using the padded data
    avg_report[target] = {
        'mean_precision': float(np.mean(precisions)),
        'std_precision': float(np.std(precisions, ddof=1)),
        'min_precision': float(np.min(precisions)),
        'max_precision': float(np.max(precisions)),
        'mean_recall': float(np.mean(recalls)),
        'std_recall': float(np.std(recalls, ddof=1)),
        'min_recall': float(np.min(recalls)),
        'max_recall': float(np.max(recalls)),
        'mean_f1-score': float(np.mean(f1_scores)),
        'std_f1-score': float(np.std(f1_scores, ddof=1)),
        'min_f1-score': float(np.min(f1_scores)),
        'max_f1-score': float(np.max(f1_scores))
      }
    
  # pprint(avg_report)
  with open(output_path, 'w') as json_file:
    json.dump(avg_report, json_file, indent=2)	

def get_ml_run_report(y_test_generalized, y_predict, y_train):
  y_train = y_train.astype(str)
  report = classification_report(y_test_generalized, y_predict, output_dict=True)
  return report

def run_model_cv(params):
  experiment_num, sample_path, stats_file_path, test_df_path, target, pipe, hierarchies_base_path = params
  train_df = pd.read_csv(sample_path, sep=';')
  test_df = pd.read_csv(test_df_path, sep=';', dtype=str)
  test_df_generalized = get_generalised(test_df, stats_file_path, hierarchies_base_path)
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
      calculate_mean_std(reports, output_paths[i], TARGETS, NUM_FOLDS)
  else:
    pipe, non_masked_pipe = pipes_tuple
    output_path = FOLDS_PATH/'ml_experiments'/f'experiment_{experiment_num}'/'classification_report.json'
    jobs = []
    for fold in range(NUM_FOLDS):
      fold_path = FOLDS_PATH/f'fold_{fold}'
      jobs.append((experiment_num, fold_path, TARGET, deepcopy(non_masked_pipe)))
    reports = parallel_run_model_cv(jobs, non_generalized=True, progressbar_desc=f'non_generalized: True')
    calculate_mean_std(reports, output_path, TARGETS, NUM_FOLDS)

def ml_worker_cv(experiment_num, strats_to_run, verbose=False):
  pipe, *_ = get_pipelines(experiment_num)
  for strat in strats_to_run:
    for k in K_LIST:
      for b in B_LIST:
        jobs = []
        avg_classification_report_output_path = OUTPUT_BASE_PATH/strat/'ml_experiments'/f'experiment_{experiment_num}'/f'k{k}'/f'B({b})'/'classification_report.json'
        for fold in range(NUM_FOLDS):
          sample_path = OUTPUT_BASE_PATH/strat/f'fold_{fold}'/f'k{k}'/f'B({b})'/f'B({b})_sample.csv'
          if sample_size_zero(sample_path=sample_path, target=TARGET):
            continue
          stats_file_path = K_ANON_BASE_PATH / f'fold_{fold}' / f'k{k}' / 'stats.json'
          test_df_path = FOLDS_PATH/f'fold_{fold}'/'test.csv'
          jobs.append((experiment_num, sample_path, stats_file_path, test_df_path, TARGET, deepcopy(pipe), HIERARCHIES_BASE_PATH))
        reports = parallel_run_model_cv(jobs, non_generalized=False, progressbar_desc=f'strat: {strat}, k: {k}, b: {b}')
        calculate_mean_std(reports, avg_classification_report_output_path, TARGETS, NUM_FOLDS)


def ml_worker_cv_ldiv(experiment_num, verbose=False):
  pipe, *_ = get_pipelines(experiment_num)
  for k in K_LIST:
    for l in L_LIST:
      jobs = []
      avg_classification_report_output_path = OUTPUT_BASE_PATH/'lDiv'/'ml_experiments'/f'experiment_{experiment_num}'/f'k{k}'/f'l{l}'/'classification_report.json'
      for fold in range(NUM_FOLDS):
        sample_path = OUTPUT_BASE_PATH/'lDiv'/f'fold_{fold}'/f'k{k}'/f'l{l}'/'sample.csv'
        stats_file_path = OUTPUT_BASE_PATH/'lDiv'/f'fold_{fold}'/f'k{k}'/f'l{l}'/'stats.json'
        if sample_size_zero(stats_file_path=stats_file_path):
          continue
        test_df_path = FOLDS_PATH/f'fold_{fold}'/'test.csv'
        jobs.append((experiment_num, sample_path, stats_file_path, test_df_path, TARGET, deepcopy(pipe), HIERARCHIES_BASE_PATH))
      reports = parallel_run_model_cv(jobs, non_generalized=False, progressbar_desc=f'strat: l-diversity, k: {k}, l: {l} ')
      calculate_mean_std(reports, avg_classification_report_output_path, TARGETS, NUM_FOLDS)

def ml_worker_cv_kanon(experiment_num, verbose=False):
  pipe, *_ = get_pipelines(experiment_num)
  for k in K_LIST:
    jobs = []
    avg_classification_report_output_path = OUTPUT_BASE_PATH/'kAnon'/'ml_experiments'/f'experiment_{experiment_num}'/f'k{k}'/'classification_report.json'
    for fold in range(NUM_FOLDS):
      sample_path = OUTPUT_BASE_PATH/'kAnon'/f'fold_{fold}'/f'k{k}'/'output_sample.csv'
      stats_file_path = OUTPUT_BASE_PATH/'kAnon'/f'fold_{fold}'/f'k{k}'/'stats.json'
      if sample_size_zero(stats_file_path=stats_file_path):
        continue
      test_df_path = FOLDS_PATH/f'fold_{fold}'/'test.csv'
      jobs.append((experiment_num, sample_path, stats_file_path, test_df_path, TARGET, deepcopy(pipe), HIERARCHIES_BASE_PATH))
    reports = parallel_run_model_cv(jobs, non_generalized=False, progressbar_desc=f'strat: k-anonymity, k: {k}')
    calculate_mean_std(reports, avg_classification_report_output_path, TARGETS, NUM_FOLDS)

def calculate_mean_std_worker(sample_strats, experiment_nums):
  for strat in sample_strats:
    for k in K_LIST:
      for b in B_LIST:
        for exp_num in experiment_nums:
          reports = []
          avg_classification_report_output_path = OUTPUT_BASE_PATH/strat/'ml_experiments'/f'experiment_{exp_num}'/f'k{k}'/f'B({b})'/'classification_report.json'
          for fold in range(NUM_FOLDS):
            report_path = OUTPUT_BASE_PATH/strat/f"fold_{fold}"/f"k{k}"/f"B({b})"/"ml_experiments"/f"experiment_{exp_num}"/"classification_report.json"
            if not os.path.isfile(report_path):
              continue # if there is no report for k,b because that dataset was empty
            with open(report_path, 'r') as json_report:
              report = json.load(json_report)
            reports.append(report)
          calculate_mean_std(reports, avg_classification_report_output_path, TARGETS, NUM_FOLDS)

if __name__ == '__main__':
  config_path = sys.argv[1]
  NUM_PROCESSES = int(sys.argv[2])
  
  # config_path = 'config/nursery.ini'
  # config_path = 'config\ACSIncome_USA_2018_binned_imbalanced_16645_acc_metric.ini'
  # config_path = 'config/cmc.ini'
  # config_path = 'config/ACSIncome_USA_2018_binned_imbalanced_16645.ini'
  # config_path = 'config/ACSIncome_USA_2018_binned_imbalanced_1664.ini'
  config_path = 'config/ACSIncome_USA_2018_binned_imbalanced_1664500.ini'
  read_config(config_path)

  # print_fully_suppressed_samples(OUTPUT_BASE_PATH/'BSAMPLE_V2', NUM_FOLDS, K_LIST, B_LIST, QID_LIST)

  # calculate_mean_std_worker(['SSAMPLE_V2', 'BSAMPLE_V2'], [1,3])

  # if RUNSAMPLEV2:
  #   sample_strats = ['SSAMPLE', 'BSAMPLE']
  #   for sample_strat in sample_strats:
  #     sample_v2 = SampleV2(OUTPUT_BASE_PATH/f'{sample_strat}', 
  #                               K_ANON_BASE_PATH, NUM_FOLDS, K_LIST, B_LIST, QID_LIST, 0.0, NUM_PROCESSES)
  #     sample_v2.run()

  if PRIVACY_METRICS:
      calculate_privacy_metrics_worker(['BSAMPLE'], journalist_risk=False, certainty=True)

  # generalization_stats(K_LIST, NUM_FOLDS, K_ANON_BASE_PATH, OUTPUT_BASE_PATH, QID_LIST)
  # sample_stats(['SSAMPLE_V2'], K_LIST, B_LIST, NUM_FOLDS, OUTPUT_BASE_PATH, QID_LIST, TARGET, TARGETS)
  # eq_per_target(['SSAMPLE_V2'], K_LIST, B_LIST, NUM_FOLDS, OUTPUT_BASE_PATH, QID_LIST, TARGET, TARGETS)
  # eq_stats(['SSAMPLE'], K_LIST, B_LIST, NUM_FOLDS, K_ANON_BASE_PATH, OUTPUT_BASE_PATH, OUTPUT_BASE_PATH/'stats', QID_LIST)
  # draw_eq_distributions(K_LIST, K_ANON_BASE_PATH, OUTPUT_BASE_PATH, QID_LIST, log_scale=True)
  # draw_combined_eq_distributions(['SSAMPLE_V2'], K_LIST, B_LIST, NUM_FOLDS, OUTPUT_BASE_PATH, K_ANON_BASE_PATH, QID_LIST)
  # print_distributions_worker(OUTPUT_BASE_PATH/'BSAMPLE_V2', NUM_FOLDS, K_LIST, B_LIST, TARGET)


  # if ML:
  #   # ml_worker_cv_nonmasked(1)
  #   # ml_worker_cv_nonmasked(3)
  #   # ml_worker_cv(1, ['SSAMPLE_V2', 'BSAMPLE_V2'])
  #   # ml_worker_cv(3, ['SSAMPLE_V2', 'BSAMPLE_V2']) 
    # ml_worker_cv_ldiv(3)
    # ml_worker_cv_kanon(3)
    

    
    
  
  
  
