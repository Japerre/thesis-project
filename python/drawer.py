import os
import sys
import json
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import ast
from sklearn.ensemble import GradientBoostingClassifier
from joblib import load
from fractions import Fraction
from pprint import pprint
from multiprocessing import Pool
from tqdm import tqdm
from fractions import Fraction


BETA = '\u03B2'
# K_LIST, B_LIST, L_LIST = None, None, None

def read_config(config_path):
	cfg = ConfigParser(interpolation=ExtendedInterpolation())
	cfg.read(config_path)
	# PATHS
	global OUTPUT_BASE_PATH, EXPERIMENT_STATS_PATH, HIERARCHIES_BASE_PATH, TRAIN_DF_PATH, K_ANON_BASE_PATH, FOLDS_PATH
	OUTPUT_BASE_PATH = Path(cfg['PATHS']['output_base_path']).resolve()
	K_ANON_BASE_PATH = OUTPUT_BASE_PATH / 'kAnon'
	EXPERIMENT_STATS_PATH = Path(cfg['PATHS']['experiment_stats_path']).resolve()
	HIERARCHIES_BASE_PATH = Path(cfg['PATHS']['hierarchies_base_path']).resolve()
	TRAIN_DF_PATH = Path(cfg['PATHS']['input_dataset_path']).resolve() / 'train.csv'
	FOLDS_PATH = Path(cfg['PATHS']['folds_path']).resolve()

	#OTHER
	global DATASET_NAME, K_LIST, B_LIST, L_LIST, NUM_FOLDS
	NUM_FOLDS = cfg.getint('VARIABLES', 'num_folds')
	DATASET_NAME = cfg['PATHS']['dataset_name']
	K_LIST, B_LIST, L_LIST = get_run_params()


def get_run_params():
  with open(EXPERIMENT_STATS_PATH, 'r') as json_file:
    stats = json.load(json_file)
  return (stats.get('k_values'), stats.get('b_values'), stats.get('l_values'))

def get_test_generalized(k_dir_name: str) -> (pd.DataFrame, pd.DataFrame):
	test_generalized_path = K_ANON_BASE_PATH / k_dir_name / 'test_generalized'
	return (pd.read_csv(test_generalized_path / 'X_test_generalized.csv', delimiter=';', decimal=','), 
				pd.read_csv(test_generalized_path / 'y_test_generalized.csv', delimiter=';', decimal=','))

def cmc_target_names():
	return {
		"1": "no use",
		"2": "long term",
		"3": "short term"
	}

def ASCIncome_target_names():
	return {
		"[0-20000[": "poor",
		"[20000-100000[": "medium",
		"[100000-inf[": "rich"
	}

def ssample_rsample_certainty(fold_number, k, b_values):
	b_values = [item for item in b_values for _ in range(2)]
	b_values_frac = [str(Fraction(item).limit_denominator()) for item in b_values]
	labels = ['RSAMPLE', 'SSAMPLE', 'RSAMPLE', 'SSAMPLE']
	certainty_paths = [OUTPUT_BASE_PATH/f'{sample_strat}/fold_{fold_number}/k{k}/b({b})/privacystats/certainty.csv' for sample_strat, b in zip(labels, b_values)]

	merged_certainties = pd.DataFrame()
	for path, x, hue in zip(certainty_paths, b_values_frac, labels):
				df = pd.read_csv(path, decimal=',', sep=';')
				df[r'$\beta$'] = x
				df['Type'] = hue
				merged_certainties = pd.concat([merged_certainties, df], ignore_index=True)
	
	ax = sns.violinplot(x=r'$\beta$', y='0', hue='Type', data=merged_certainties, scale_hue=True, cut=0, scale='area')
	ax.legend()
	ax.xaxis.set_label_coords(0, -0.02)
	plt.ylabel('certainty')
	plt.tight_layout()
	plt.show()
	

def compare_certainty_plots(fold_number, k, b, sample_strats, sample_strats_names):
	rcParams.update({'font.size': 14})
	certainty_path_1 = OUTPUT_BASE_PATH/f'{sample_strats[0]}/fold_{fold_number}/k{k}/b({b})/privacystats/certainty.csv'
	certainty_path_2 = OUTPUT_BASE_PATH/f'{sample_strats[1]}/fold_{fold_number}/k{k}/b({b})/privacystats/certainty.csv'

	certainty_data_1 = pd.read_csv(certainty_path_1, sep=';', decimal=',')
	certainty_data_2 = pd.read_csv(certainty_path_2, sep=';', decimal=',')

	# Combine data into one DataFrame with a 'version' column
	certainty_data_1['version'] = sample_strats_names[0]
	certainty_data_2['version'] = sample_strats_names[1]
	combined_data = pd.concat([certainty_data_1, certainty_data_2])

	# Create the violin plot with seaborn
	sns.violinplot(x='version', y='0', data=combined_data, cut=0)
	plt.xlabel('')
	plt.ylabel('certainty')
	# plt.title(f'{DATASET_NAME} - stratified balanced sampling voor k={k}, {BETA}={b}')
	# plt.tight_layout()
	# Save the plot
	plt.show()

	
def privacy_plot(params):
	privacy_metric_path, privacy_metric_output_plot_path = params
	data = pd.read_csv(privacy_metric_path, sep=';', decimal=',')
	sns.violinplot(y=data['0'], cut=0)
	plt.savefig(privacy_metric_output_plot_path, bbox_inches='tight', dpi=300)
	plt.close()

def privacy_plots_worker(sample_strats: list, certainty=False, journalist_risk=False):
	jobs = []
	for sample_strat in sample_strats:
		sample_strat_base_path = OUTPUT_BASE_PATH/sample_strat
		privacy_folder_paths = []
		for foldername, subfolders, filenames in os.walk(sample_strat_base_path):
			if(Path(foldername).name == 'privacystats'):
				privacy_folder_paths.append(foldername)
		for folder in privacy_folder_paths:
			if certainty:
				certainty_path = os.path.join(folder, "certainty.csv")
				certainty_plot_path = os.path.join(folder, "certainty.png")
				jobs.append((certainty_path, certainty_plot_path))
			if journalist_risk:
				journalist_risk_path = os.path.join(folder, "journalistRisk.csv")
				journalist_risk_plot_path = os.path.join(folder, "journalistRisk.png")
				jobs.append((journalist_risk_path, journalist_risk_plot_path))
	with Pool(processes=NUM_PROCESSES) as pool:
	  return list(tqdm(pool.imap(privacy_plot, jobs),total=len(jobs),desc='plotting privacy plots'))

def compare_violin_plots(fold, k, metric, sample_strat, title=None):
	rcParams.update({'font.size': 14})
	all_data = pd.DataFrame()

	# Collect data
	for b in B_LIST:
			base_path = OUTPUT_BASE_PATH / sample_strat / f'fold_{fold}' / f'k{k}'
			privacy_metric_path = base_path / f'B({b})' / 'privacystats' / f'{metric}.csv'
			df = pd.read_csv(privacy_metric_path, decimal=',', sep=';')
			df['B'] = b  # Add a column to indicate the B value for this dataset
			all_data = pd.concat([all_data, df], ignore_index=True)

	# Plotting		
	plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
	sns.violinplot(x='B', y='0', data=all_data, cut=0, order=reversed(sorted(all_data['B'].unique())))  # Replace 'your_column_name' with the name of the column you want to plot

	labels = [item.get_text() for item in plt.gca().get_xticklabels()]
	frac_labels = [str(Fraction(float(label)).limit_denominator()) for label in labels]

	if metric == 'journalistRisk':
		plt.ylabel('Journalist Risk')
	if metric == 'certainty':
		plt.ylabel('Certainty')
	if title is not None:
		plt.title(title)
	# else:
	# 	plt.title(f'Violin Plot of {metric} by B Value for {sample_strat}, k={k}, fold={fold}')
	
	plt.xticks(range(len(frac_labels)), frac_labels)
	plt.xlabel(BETA, loc='left', labelpad=-12)
	plt.show()	


def grouped_bar_chart(sample_strats: list, experiment_num, plot_non_masked: bool):
	for sample_strat in sample_strats:
		# sample_strat = 'SSample'
		sample_dir = OUTPUT_BASE_PATH / sample_strat
		output_dir = OUTPUT_BASE_PATH / 'ml_plots' / f'experiment_{experiment_num}' / sample_strat / 'grouped_bar_chart'
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		df, avg_strats = classification_report_json_to_df(sample_dir, experiment_num)

		metrics = ['precision', 'recall', 'f1-score']

		b_values = pd.unique(df['b']).tolist()
		k_values = pd.unique(df['k']).tolist()
		b_values.sort(reverse=True)
		k_values.sort()
		
		fraction_b_values = [Fraction(value).limit_denominator() for value in b_values]

		for avg_strat in avg_strats:
			for metric in metrics:
				my_dict = {}
				for k in k_values:
					my_dict[k] = df.query(f'k == {k}')[f'{avg_strat}.{metric}'].tolist()
					my_dict[k].sort(reverse=True)
					# print(my_dict)
				
				x = np.arange(len(b_values))
				width = 0.13
				multiplier = 0

				plt.figure()

				for key, value in my_dict.items():
					offset = width * multiplier
					rects = plt.bar(x + offset, value, width, label=f'k={key}')
					multiplier += 1

				if plot_non_masked:
					non_masked_report_path = OUTPUT_BASE_PATH / 'inputDataset' / 'ml_experiments' / f'experiment_{experiment_num}' / 'classification_report.json'
					with open(non_masked_report_path, 'r') as f:
						non_masked_report = json.load(f)
					y_val = non_masked_report[avg_strat][metric]
					plt.axhline(y=y_val, label='non masked', color='red', linestyle='--')

				plt.title(f'{avg_strat} {metric} for {sample_strat}')
				plt.xlabel(BETA, loc='left', labelpad=-9)
				plt.legend(loc='lower right', shadow=True)
				plt.xticks(ticks=x+float((multiplier-1)/2)*width, labels=fraction_b_values)
				
				plt.savefig(output_dir/f'{avg_strat}_{metric}.png', bbox_inches='tight')
				plt.close()

def empty_samples_heatmap():
	ldiv_base_path = OUTPUT_BASE_PATH/'ldiv'
	empty_samples_matrix = np.zeros((len(K_LIST), len(L_LIST))).astype(int)
	for k_index, k in enumerate(K_LIST):
		for l_index, l in enumerate(L_LIST):
			for fold in range(NUM_FOLDS):
				stats_path = ldiv_base_path/f'fold_{fold}/k{k}/l{l}/stats.json'
				with open(stats_path, 'r') as stats_file:
					stats = json.load(stats_file)
				if stats.get('sample size', 1) == 0:
					empty_samples_matrix[k_index, l_index] +=1
					
	# Plotting the heatmap
	plt.figure(figsize=(10, 8))
	sns.heatmap(empty_samples_matrix, annot=True, fmt="d", cmap="viridis",
							xticklabels=L_LIST, yticklabels=K_LIST)
	plt.xlabel('L values')
	plt.ylabel('K values')
	plt.title('Heatmap of Empty Sample Sizes Across Folds')
	plt.show()
	
def suppressed_heatmap(sample_strat):
	sample_base_path = OUTPUT_BASE_PATH/sample_strat
	if sample_strat == 'lDiv':
		l_div = True
		b_or_l_list = L_LIST
	else:
		l_div = False
		b_or_l_list = B_LIST
	
	suppressed_matrix = np.zeros((len(K_LIST), len(L_LIST))).astype(int)
	for k_index, k in enumerate(K_LIST):
		for b_l_index, b_l in enumerate(b_or_l_list):
			for fold in range(NUM_FOLDS):
				suppressed_records_list = []
				if l_div:
					stats_path = sample_base_path/f'fold_{fold}/k{k}/l{b_l}/stats.json'
				else:
					stats_path = sample_base_path/f'fold_{fold}/k{k}/B({b_l})/stats.json'
				with open(stats_path, 'r') as stats_file:
					stats = json.load(stats_file)
				suppressed_records_list.append(stats.get('suppressed in sample', 1))
			suppressed_matrix[k_index, b_l_index] = np.mean(suppressed_records_list)
	
	# Plotting the heatmap
	plt.figure(figsize=(10, 8))
	sns.heatmap(suppressed_matrix, annot=True, fmt="d", cmap="viridis",
							xticklabels=L_LIST, yticklabels=K_LIST)
	plt.xlabel('L values')
	plt.ylabel('K values')
	plt.title('Heatmap of suppressed records Sizes Across Folds')
	plt.show()


def figure_8():
	# dataset = 'ACSIncome_USA_2018_binned_imbalanced_1664500'
	# dataset = 'ACSIncome_USA_2018_binned_imbalanced_16645'
	dataset = 'ACSIncome_USA_2018_binned_imbalanced_1664'
	# dataset = 'nursery'
	# dataset = 'cmc'
	base_dir = Path(f'../data/output2/{dataset}').resolve()
	strats = ['BSAMPLE_V2', 'lDiv']
	strat_names = ['bsample_s', 'l-diversity']
	ml_dirs = []
	kanon_dirs = []
	report_paths = []
	non_masked_report_paths = []
	
	# score_types = ['macro avg', 'very_recom']
	# score_types = ['macro avg', '2']
	score_types = ['macro avg', '[100000-inf[']
	# names = ['macro avg', 'long-term']
	# names = ['macro avg', 'very_recom']
	names = ['macro avg', 'rich']
	k_list = [10, 20, 50, 100]
	b_list = [0.5, 0.2, 0.01]
	l_list = [2.0, 2.5, 3.0]
	# l_list = [2.0, 3.0, 4.0]
	data_frames = []
	fraction_b_values = [Fraction(value).limit_denominator() for value in b_list]

	for i, strat in enumerate(strats):
		ml_dir = base_dir/f'{strat}/ml_experiments/experiment_3'
		non_masked_report_path = base_dir/'inputdataset/folds/ml_experiments/experiment_1/classification_report.json'
		ml_dirs.append(ml_dir)
		non_masked_report_paths.append(non_masked_report_path)
		if i == 0:
			df, _ = classification_report_json_to_df(ml_dir, k_list, b_list, ldiv=False)
		else:
			df, _ = classification_report_json_to_df(ml_dir, k_list, b_list=None, ldiv=True, l_list=l_list)
		data_frames.append(df)

	fig, axes = plt.subplots(2, 2, figsize=(10, 5), sharey=True)
	plt.subplots_adjust(hspace=0.5, wspace=0.1)

	for i, score_type in enumerate(score_types):
		for j, (df, non_masked_report_path) in enumerate(zip(data_frames, non_masked_report_paths)):

			with open(non_masked_report_path, 'r') as f:
				non_masked_report = json.load(f)
				y_val = non_masked_report[score_type]['mean_f1-score']
				axes[i,j].axhline(y=y_val, label='originele\n dataset', color='red', linestyle='--', zorder=0)

			for k in k_list:
				if j == 0:
					k_rows = df.query(f'k == {k} and b in @b_list').sort_values(by='b', ascending=False)
				else:
					k_rows = df.query(f'k == {k} and l in @l_list').sort_values(by='l', ascending=True)
				values = k_rows[f'{score_type}.mean_f1-score'].tolist()
				min_values = k_rows[f'{score_type}.min_f1-score'].tolist()
				max_values = k_rows[f'{score_type}.max_f1-score'].tolist()
				lower_errors = np.array(values) - np.array(min_values)
				upper_errors = np.array(max_values) - np.array(values)
				error_bars = [lower_errors, upper_errors]
				x = np.arange(len(b_list))
				width = 0.13
				offset = width * k_list.index(k)
				axes[i,j].bar(x + offset, values, width, yerr=error_bars, label=f'k={k}')

			axes[i,j].set_title(f'{strat_names[j]} - {names[i]}')
			if j == 0:
				axes[i,j].set_xlabel(BETA, loc='left', labelpad=-9)
				axes[i,j].set_xticks(ticks=x+float((len(k_list)-1)/2)*width)
				axes[i,j].set_xticklabels([str(b) for b in fraction_b_values])
			else:
				axes[i,j].set_xlabel('l', loc='left', labelpad=-9)
				axes[i,j].set_xticks(ticks=x+float((len(k_list)-1)/2)*width)
				axes[i,j].set_xticklabels(l_list)

	# plt.subplots_adjust(right=0.8)		
	handles, labels = axes[0, 0].get_legend_handles_labels()
	fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1,1))
	# plt.tight_layout()
	plt.show()	


def figure_6():
	base_dir = Path('../data/output2').resolve()
	sizes = ['1664500', '16645', '1664']
	sample_strat = 'BSAMPLE_V2'
	experiment_num = 3
	# sizes = ['16645',  '1664']
	dataset_names = [f'ACSIncome {size}' for size in sizes ]
	ml_dirs = []
	kanon_dirs = []
	report_paths = []
	non_masked_report_paths = []

	# Constants and settings
	k_list = [10, 20, 50, 100]
	b_list = [0.5, 0.1, 0.01]
	data_frames = []

	# Generate paths for each size
	for size in sizes:
		size_dir = base_dir / f'ACSIncome_USA_2018_binned_imbalanced_{size}'
		
		ml_dir = size_dir / f'{sample_strat}/ml_experiments/experiment_{experiment_num}'
		kanon_dir = size_dir / f'kAnon/ml_experiments/experiment_{experiment_num}'
		report_path = size_dir / f'inputDataset/folds/ml_experiments/experiment_{experiment_num}/classification_report.json'
		non_masked_report_path = size_dir / f'inputdataset/folds/ml_experiments/experiment_{experiment_num}/balanced/classification_report.json'
		
		ml_dirs.append(ml_dir)
		kanon_dirs.append(kanon_dir)
		report_paths.append(report_path)
		non_masked_report_paths.append(non_masked_report_path)

	# Load data from classification reports
	for ml_dir, kanon_dir in zip(ml_dirs, kanon_dirs):
		df, _ = classification_report_json_to_df(ml_dir, k_list, b_list)  # Adjust this function to return a DataFrame
		
		# Append k-anonymized data with b=1
		for k in k_list:
			with open(kanon_dir / f'k{k}/classification_report.json', 'r') as json_file:
				report = json.load(json_file)
				flat_json = pd.json_normalize(report)
				flat_json['k'] = k
				flat_json['b'] = 1
				df = pd.concat([df, flat_json], ignore_index=True)
				
		data_frames.append(df)

	b_list.insert(0,1)
	fraction_b_values = [Fraction(value).limit_denominator() for value in b_list]

	# Setup plot
	score_types = ['macro avg', '[100000-inf[']
	names = ['macro avg f1-score', 'rich f1-score']
	fig, axes = plt.subplots(len(sizes), len(score_types), figsize=(10, 5), sharey=True)
	plt.subplots_adjust(hspace=0.5, wspace=0.1)

	for i, score_type in enumerate(score_types):
		for j, (df, report_path, non_masked_report_path) in enumerate(zip(data_frames, report_paths, non_masked_report_paths)):
			# Plot each k value as a bar group
			for k in k_list:
				k_rows = df.query(f'k == {k} and b in @b_list').sort_values(by='b', ascending=False)
				values = k_rows[f'{score_type}.mean_f1-score'].tolist()
				min_values = k_rows[f'{score_type}.min_f1-score'].tolist()
				max_values = k_rows[f'{score_type}.max_f1-score'].tolist()
				lower_errors = np.array(values) - np.array(min_values)
				upper_errors = np.array(max_values) - np.array(values)
				error_bars = [lower_errors, upper_errors]
				x = np.arange(len(b_list))
				width = 0.13
				offset = width * k_list.index(k)
				axes[j,i].bar(x + offset, values, width, yerr=error_bars, label=f'k={k}', capsize=4)

			with open(non_masked_report_path, 'r') as f:
				non_masked_report = json.load(f)
				y_val = non_masked_report[score_type]['mean_f1-score']
				axes[j,i].axhline(y=y_val, label='Originele Dataset\n(Gebalanceerd)', color='red', linestyle='--', zorder=0)

			axes[j,i].set_title(f'{dataset_names[j]} - {names[i]}')
			axes[j,i].set_xlabel(BETA, loc='left', labelpad=-9)
			axes[j,i].set_xticks(ticks=x+float((len(k_list)-1)/2)*width)
			axes[j,i].set_xticklabels([str(b) for b in fraction_b_values])

	# plt.subplots_adjust(right=0.8)		
	handles, labels = axes[0, 0].get_legend_handles_labels()
	fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1,1))
	# plt.tight_layout()
	plt.show()


def figure_5():
	ACSIncome_big_ml_experiments_dir = Path(r'../data/output2/ACSIncome_USA_2018_binned_imbalanced_1664500/SSAMPLE_V2/ml_experiments/experiment_1').resolve()
	ACSIncome_big_non_masked_report_path = Path(r'../data/output2/ACSIncome_USA_2018_binned_imbalanced_1664500/inputDataset/folds/ml_experiments/experiment_1/classification_report.json').resolve()
	ACSIncome_small_ml_experiments_dir = Path(r'../data/output2/ACSIncome_USA_2018_binned_imbalanced_1664/SSAMPLE_V2/ml_experiments/experiment_1').resolve()
	ACSIncome_small_non_masked_report_path = Path(r'../data/output2/ACSIncome_USA_2018_binned_imbalanced_1664/inputDataset/folds/ml_experiments/experiment_1/classification_report.json').resolve()
	ACSIncome_big_kanon_ml_experiments_dir = Path(r'../data/output2/ACSIncome_USA_2018_binned_imbalanced_1664500/kAnon/ml_experiments/experiment_1').resolve()
	ACSIncome_small_kanon_ml_experiments_dir = Path(r'../data/output2/ACSIncome_USA_2018_binned_imbalanced_1664/kAnon/ml_experiments/experiment_1').resolve()

	k_list = K_LIST
	b_list = [0.5, 0.01]
	
	big_df, _ = classification_report_json_to_df(ACSIncome_big_ml_experiments_dir, k_list, b_list)
	small_df, _ = classification_report_json_to_df(ACSIncome_small_ml_experiments_dir, k_list, b_list)

	b_list.insert(0,1)
	fraction_b_values = [Fraction(value).limit_denominator() for value in b_list]
	data_frames = [big_df, small_df]
	kanon_ml_experiments_dirs = [ACSIncome_big_kanon_ml_experiments_dir, ACSIncome_small_kanon_ml_experiments_dir]
	
	for i, (df, kanon_experiments_path) in enumerate(zip(data_frames, kanon_ml_experiments_dirs)):
		for k in k_list:
			with open(kanon_experiments_path/f'k{k}/classification_report.json', 'r') as json_file:
				report = json.load(json_file)
				flat_json = pd.json_normalize(report)
				flat_json['k'] = k
				flat_json['b'] = 1
			df = pd.concat([df, flat_json], ignore_index=True)
			# print(df['b'].unique())
		data_frames[i] = df
		
	non_masked_reports_paths = [ACSIncome_big_non_masked_report_path, ACSIncome_small_non_masked_report_path]
	fig, axes = plt.subplots(1, 2, figsize=(12, 6))
	
	y_min = float('inf')
	y_max = float('-inf')
	
	# First pass to determine the global y-axis limits
	for df in data_frames:
		# print(df['b'].unique())
		# print(f'blist = {b_list}')
		for k in k_list:
			k_rows = df.query(f'k == {k} and b in @b_list').sort_values(by='b', ascending=False)
			# print(k_rows['b'].unique)
			min_values = k_rows['[100000-inf[.min_f1-score'].tolist()
			max_values = k_rows['[100000-inf[.max_f1-score'].tolist()
			y_min = min(y_min, min(min_values))
			y_max = max(y_max, max(max_values))
	
	# Second pass to plot
	handles, labels = [], []
	
	for i, df in enumerate(data_frames):
		width = 0.13
		x = np.arange(len(b_list))
		
		non_masked_report_path = non_masked_reports_paths[i]
		with open(non_masked_report_path, 'r') as f:
			non_masked_report = json.load(f)
		y_val = non_masked_report['[100000-inf[']['mean_f1-score']
		non_masked_line = axes[i].axhline(y=y_val, label='originele\ndataset', color='red', linestyle='--')
		if i == 0:  # Add the non-masked line handle to the legend only once
			handles.append(non_masked_line)
			labels.append('originele\ndataset')

		for k in k_list:
			k_rows = df.query(f'k == {k} and b in @b_list').sort_values(by='b', ascending=False)
			value = k_rows['[100000-inf[.mean_f1-score'].tolist()
			min_values = k_rows['[100000-inf[.min_f1-score'].tolist()
			max_values = k_rows['[100000-inf[.max_f1-score'].tolist()
			lower_errors = np.array(value) - np.array(min_values)
			upper_errors = np.array(max_values) - np.array(value)
			error_bars = [lower_errors, upper_errors]
			
			offset = width * (k_list.index(k))
			rects = axes[i].bar(x + offset, value, width, label=f'k={k}', yerr=error_bars, capsize=5)

			if i == 0:
				handles.append(rects[0])
				labels.append(f'k={k}')
		
		axes[i].set_title(f"ACSIncome {'1664500' if i == 0 else '1644'}")
		axes[i].set_xticks(x + width * (len(k_list) - 1) / 2)
		axes[i].set_xlabel(BETA, loc='left', labelpad=-9)
		axes[i].set_xticklabels([b for b in fraction_b_values])
		axes[i].set_ylim([y_min, y_max])  # Set the same y-axis limits for both subplots

	fig.legend(handles, labels, loc='upper right', ncol=1)
	plt.tight_layout()
	plt.show()



def figure_1():
	ACSIncome_big_ml_experiments_dir = Path(r'../data/output2/ACSIncome_USA_2018_binned_imbalanced_1664500/SSAMPLE_V2/ml_experiments/experiment_1')
	ACSIncome_big_non_masked_report_path = Path(r'../data/output2/ACSIncome_USA_2018_binned_imbalanced_1664500/inputDataset/folds/ml_experiments/experiment_1/classification_report.json')
	ACSIncome_small_ml_experiments_dir = Path(r'../data/output2/ACSIncome_USA_2018_binned_imbalanced_1664/SSAMPLE_V2/ml_experiments/experiment_1')
	ACSIncome_small_non_masked_report_path = Path(r'../data/output2/ACSIncome_USA_2018_binned_imbalanced_1664/inputDataset/folds/ml_experiments/experiment_1/classification_report.json')

	k_list = K_LIST
	b_list = B_LIST
	fraction_b_values = [Fraction(value).limit_denominator() for value in b_list]
	
	big_df, _ = classification_report_json_to_df(ACSIncome_big_ml_experiments_dir)
	small_df, _ = classification_report_json_to_df(ACSIncome_small_ml_experiments_dir)
	
	data_frames = [big_df, small_df]
	non_masked_reports_paths = [ACSIncome_big_non_masked_report_path, ACSIncome_small_non_masked_report_path]
	fig, axes = plt.subplots(1, 2, figsize=(12, 6))
	
	y_min = float('inf')
	y_max = float('-inf')
	
	# First pass to determine the global y-axis limits
	for df in data_frames:
		for k in k_list:
			k_rows = df.query(f'k == {k} and b in @b_list').sort_values(by='b', ascending=False)
			min_values = k_rows['[100000-inf[.min_f1-score'].tolist()
			max_values = k_rows['[100000-inf[.max_f1-score'].tolist()
			y_min = min(y_min, min(min_values))
			y_max = max(y_max, max(max_values))
	
	# Second pass to plot
	handles, labels = [], []
	for i, df in enumerate(data_frames):
		width = 0.18
		x = np.arange(len(b_list))
		
		non_masked_report_path = non_masked_reports_paths[i]
		with open(non_masked_report_path, 'r') as f:
			non_masked_report = json.load(f)
		y_val = non_masked_report['[100000-inf[']['mean_f1-score']
		non_masked_line = axes[i].axhline(y=y_val, label='non masked', color='red', linestyle='--')

		for k in k_list:
			k_rows = df.query(f'k == {k} and b in @b_list').sort_values(by='b', ascending=False)
			value = k_rows['[100000-inf[.mean_f1-score'].tolist()
			min_values = k_rows['[100000-inf[.min_f1-score'].tolist()
			max_values = k_rows['[100000-inf[.max_f1-score'].tolist()
			lower_errors = np.array(value) - np.array(min_values)
			upper_errors = np.array(max_values) - np.array(value)
			error_bars = [lower_errors, upper_errors]
			
			offset = width * (k_list.index(k))
			rects = axes[i].bar(x + offset, value, width, label=f'k={k}', yerr=error_bars, capsize=5)

			if i == 0:
				handles.append(rects[0])
				labels.append(f'k={k}')
		
		if i == 0:  # Add the non-masked line handle to the legend only once
			handles.append(non_masked_line)
			labels.append('non masked')
    
		axes[i].set_title(f"ACSIncome {'1664500' if i == 0 else '1644'}")
		axes[i].set_xticks(x + width * (len(k_list) - 1) / 2)
		axes[i].set_xlabel(BETA, loc='left', labelpad=-9)
		axes[i].set_xticklabels([b for b in fraction_b_values])
		axes[i].set_ylim([y_min, y_max])  # Set the same y-axis limits for both subplots

	fig.legend(handles, labels, loc='upper right', ncol=1)
	plt.tight_layout()
	plt.show()



def grouped_bar_chart_big_image(
		sample_strats: list, 
		experiment_num,
		title=None,
		target_translation_dict=None, 
		plot_non_masked=True,
		plot_std=False, 
		plot_range=False,
		ros=False,
		rus=False,
		ldiv=False,
		include_kanon=False,
		k_list = None,
		b_list = None,
		l_list = None
		):
	
	if not k_list:
		k_list = K_LIST
	if not b_list:
		b_list = B_LIST
	if not l_list:
		l_list = L_LIST

	for sample_strat in sample_strats:
		ml_experiments_dir = OUTPUT_BASE_PATH/sample_strat/'ml_experiments'/f'experiment_{experiment_num}'
		output_dir = OUTPUT_BASE_PATH / 'ml_plots' / f'experiment_{experiment_num}' / sample_strat / 'grouped_bar_chart'
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		df, avg_strats = classification_report_json_to_df(ml_experiments_dir, k_list, b_list, ldiv=ldiv, l_list=l_list)
		if include_kanon:
			kanon_dir = OUTPUT_BASE_PATH/'kAnon'/'ml_experiments'/f'experiment_{experiment_num}'
			for k in k_list:
				with open(kanon_dir/f'k{k}'/'classification_report.json', 'r') as json_file:
					report = json.load(json_file)
					flat_json = pd.json_normalize(report)
					flat_json['k'] = k
					flat_json['b'] = 1
				df = pd.concat([df, flat_json], ignore_index=True)

		metrics = ['precision', 'recall', 'f1-score']
		if include_kanon:
			b_list.insert(0, 1)
		fraction_b_values = [Fraction(value).limit_denominator() for value in b_list]
	
		# Create a single figure for all subplots
		fig, axes = plt.subplots(len(avg_strats), len(metrics), figsize=(15, 10))
		fig.suptitle(f'{DATASET_NAME} {sample_strat}', fontsize=16) if title is None else fig.suptitle(title, fontsize=16)
		
		for i, avg_strat in enumerate(avg_strats):
			if avg_strat not in ['macro avg', 'weighted avg'] and target_translation_dict is not None:
				target_name = target_translation_dict[avg_strat]
			else:
				target_name = avg_strat
			for j, metric in enumerate(metrics):
				mean_dict, std_dict, min_dict, max_dict = {}, {}, {}, {}

				if ldiv:
					for k in k_list:
						k_rows = df.query(f'k == {k} and l in @l_list').sort_values(by='l', ascending=True)
						mean_dict[k] = k_rows[f'{avg_strat}.mean_{metric}'].tolist()
						std_dict[k] = k_rows[f'{avg_strat}.std_{metric}'].tolist()
						min_dict[k] = k_rows[f'{avg_strat}.min_{metric}'].tolist()
						max_dict[k] = k_rows[f'{avg_strat}.max_{metric}'].tolist()
					x = np.arange(len(L_LIST))
				
				if not ldiv:
					for k in k_list:
						k_rows = df.query(f'k == {k} and b in @b_list').sort_values(by='b', ascending=False)
						mean_dict[k] = k_rows[f'{avg_strat}.mean_{metric}'].tolist()
						std_dict[k] = k_rows[f'{avg_strat}.std_{metric}'].tolist()
						min_dict[k] = k_rows[f'{avg_strat}.min_{metric}'].tolist()
						max_dict[k] = k_rows[f'{avg_strat}.max_{metric}'].tolist()
					x = np.arange(len(fraction_b_values))
				
				width = 0.13
				multiplier = 0

				for key, value in mean_dict.items():
					# print(f'key: {key}, value: {value}')
					offset = width * multiplier
					if plot_std:
						std_values = std_dict[key]
						rects = axes[i, j].bar(x + offset, value, width, label=f'k={key}', yerr=std_values)
					elif plot_range:
						min_values = min_dict[key]
						max_values = max_dict[key]
						lower_errors = [mean - min_val for mean, min_val in zip(value, min_values)]
						upper_errors = [max_val - mean for mean, max_val in zip(value, max_values)]
						error_bars = [lower_errors, upper_errors]
						rects = axes[i, j].bar(x + offset, value, width, label=f'k={key}', yerr=error_bars)
					else:
						rects = axes[i, j].bar(x + offset, value, width, label=f'k={key}')
					multiplier += 1

				if rus or ros:
					label = 'non-masked\nbalanced ROS' if ros else 'non-masked\nbalanced RUS'
					non_masked_report_path = FOLDS_PATH/'ml_experiments'/f'experiment_{experiment_num}'/'non_balanced'/'classification_report.json'
					non_masked_balanced_path = FOLDS_PATH/'ml_experiments'/f'experiment_{experiment_num}'/'balanced'/'classification_report.json'
					with open(non_masked_balanced_path, 'r') as f:
						non_masked_balanced_report = json.load(f)
					y_val = non_masked_balanced_report[avg_strat][f'mean_{metric}']
					axes[i, j].axhline(y=y_val, label=label, color='black', linestyle='--')
				else:
					non_masked_report_path = FOLDS_PATH/'ml_experiments'/f'experiment_{experiment_num}'/'classification_report.json'
				
				if plot_non_masked:
					with open(non_masked_report_path, 'r') as f:
						non_masked_report = json.load(f)
					y_val = non_masked_report[avg_strat][f'mean_{metric}']
					axes[i, j].axhline(y=y_val, label='non masked', color='red', linestyle='--')
				
				axes[i, j].set_title(f'{target_name} - {metric}')
				if not ldiv:
					axes[i, j].set_xlabel(BETA, loc='left', labelpad=-9)
					axes[i, j].set_xticklabels(fraction_b_values)
				if ldiv:
					axes[i, j].set_xlabel('l', loc='left', labelpad=-9)
					axes[i, j].set_xticklabels(L_LIST)
				# axes[i, j].set_ylabel('metric score')
				axes[i, j].set_xticks(ticks=x+float((multiplier-1)/2)*width)

		# Add a single legend for the entire figure
		handles, labels = axes[0, 0].get_legend_handles_labels()
		fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 1))

		# Save the figure
		plt.tight_layout()
		plt.savefig(output_dir / 'grouped_bar_charts.png', bbox_inches='tight')
		plt.close()


def classification_report_json_to_df(ml_experiments_dir: Path, k_list, b_list, ldiv=False, l_list=None):
	df = pd.DataFrame()
	for k in k_list:
		if not ldiv:
			for b in b_list:
				with open(ml_experiments_dir/f'k{k}/B({b})/classification_report.json', 'r') as json_file:
					report = json.load(json_file)
					flat_json = pd.json_normalize(report)
					flat_json['k'] = k
					flat_json['b'] = b
				df = pd.concat([df, flat_json], ignore_index=True)
		if ldiv:
			for l in l_list:
				with open(ml_experiments_dir/f'k{k}/l{l}/classification_report.json', 'r') as json_file:
					report = json.load(json_file)
					flat_json = pd.json_normalize(report)
					flat_json['k'] = k
					flat_json['l'] = l
				df = pd.concat([df, flat_json], ignore_index=True)
	
	df.fillna(0, inplace=True)
	avg_strats = [key for key in report.keys() if key not in ['accuracy', 'train_record_counts']]
	return (df, avg_strats)


def cmp_balancing_pre_post(
		title=None,
		target_translation_dict=None, 
		):
	
	BSample_dir = OUTPUT_BASE_PATH/'BSAMPLE_V2'/'ml_experiments'/'experiment_1'
	SSample_dir = OUTPUT_BASE_PATH/'SSAMPLE_V2'/'ml_experiments'/'experiment_3'

	output_dir = OUTPUT_BASE_PATH / 'ml_plots' / 'experiment_1,3' / 'diff_pre_post_balancing'
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	ssample_df, avg_strats = classification_report_json_to_df(SSample_dir)
	bsample_df, avg_strats = classification_report_json_to_df(BSample_dir)
	
	metrics = ['precision', 'recall', 'f1-score']

	fraction_b_values = [Fraction(value).limit_denominator() for value in B_LIST]

	# Create a single figure for all subplots
	fig, axes = plt.subplots(len(avg_strats), len(metrics), figsize=(15, 10))
	fig.suptitle(title, fontsize=16)
	
	for i, avg_strat in enumerate(avg_strats):
		if avg_strat not in ['macro avg', 'weighted avg'] and target_translation_dict is not None:
			target_name = target_translation_dict[avg_strat]
		else:
			target_name = avg_strat
		for j, metric in enumerate(metrics):
			ssample_dict = {}
			bsample_dict = {}
			for k in K_LIST:
				ssample_dict[k] = ssample_df.query(f'k == {k}').sort_values(by='b', ascending=False)[f'{avg_strat}.mean_{metric}'].tolist()
				bsample_dict[k] = bsample_df.query(f'k == {k}').sort_values(by='b', ascending=False)[f'{avg_strat}.mean_{metric}'].tolist()
			
			x = np.arange(len(B_LIST))
			width = 0.13
			multiplier = 0

			for k in K_LIST:
				offset = width * multiplier
				bar_value = np.array(bsample_dict[k]) - np.array(ssample_dict[k])
				rects = axes[i, j].bar(x + offset, bar_value, width, label=f'k={k}')
				multiplier += 1

			axes[i, j].set_title(f'{target_name} - {metric}')
			axes[i, j].set_xlabel('sampling rate')
			# axes[i, j].set_ylabel('metric score')
			axes[i, j].set_xticks(ticks=x+float((multiplier-1)/2)*width)
			axes[i, j].set_xticklabels(fraction_b_values)

	# Add a single legend for the entire figure
	handles, labels = axes[0, 0].get_legend_handles_labels()
	fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 1))

	# Save the figure
	plt.tight_layout()
	# plt.show()
	plt.savefig(output_dir / 'diff_pre_post_balancing.png', bbox_inches='tight')
	plt.close()


NUM_PROCESSES = None

if __name__ == '__main__':
	NUM_PROCESSES = int(sys.argv[1])
	# config_path = 'config/cmc.ini'
	# config_path = 'config/nursery.ini'
	# config_path = 'config\ACSIncome_USA_2018_binned_imbalanced_16645_acc_metric.ini'
	# config_path = 'config/ACSIncome_USA_2018_binned_imbalanced_16645.ini'
	# config_path = 'config/ACSIncome_USA_2018_binned_imbalanced_1664.ini'
	config_path = 'config/ACSIncome_USA_2018_binned_imbalanced_1664500.ini'
	read_config(config_path)
	# compare_violin_plots(0, 5, 'journalistRisk', 'SSAMPLE_V2')
	# figure_6()
	# figure_6()
	# privacy_plots_worker(['SSAMPLE_V2'], certainty=True, journalist_risk=False)
	# ssample_rsample_certainty(1, 10, [0.25, 0.0625])
	# empty_samples_heatmap()
	# suppressed_heatmap('lDiv')
	compare_certainty_plots(0, 10, 0.5, ['BSAMPLE', 'BSAMPLE_V2'], ['bsample_BE', 'bsample_S'])
	# grouped_bar_chart_big_image(['lDiv'], 1, ldiv=True, plot_range=True, target_translation_dict=ASCIncome_target_names())
	# compare_violin_plots(0, 5, 'journalistRisk', 'SSAMPLE', title=f'Journalist Risk voor k=5 bij dalende {BETA}')
	# grouped_bar_chart_big_image(['SSAMPLE_V2'], experiment_num=1, plot_range=True, include_kanon=True,  k_list=K_LIST, b_list=B_LIST)
	# grouped_bar_chart_big_image(['lDiv'], 1, ldiv=True, plot_range=True)
	# grouped_bar_chart_big_image(['SSAMPLE'], 3, target_translation_dict=ASCIncome_target_names(), rus=True, title='ASCIncome RUS balancing after SSample', plot_std=True)
	# grouped_bar_chart_big_image(
	# 	sample_strats=['SSAMPLE', 'BSAMPLE', 'RSAMPLE'], 
	# 	experiment_num=1,
	# 	target_translation_dict=ASCIncome_target_names(), 
	# 	# rus=True,
	# 	# title="ASCIncome RUS balancing after SSample"
	# 	)
	# cmp_balancing_pre_post(title="nursery BSAMPLE_V2 vs SSample_V2 with RUS in ML pipeline")
	# grouped_bar_chart_big_image(['SSAMPLE'], 3, target_translation_dict=ASCIncome_target_names(), rus=True, title='ASCIncome RUS balancing after SSample')
	# grouped_bar_chart_big_image(['SSAMPLE'], 3, target_translation_dict=cmc_target_names(), ros=True, title='CMC ROS balancing after SSample', plot_std=True)
	

	
	