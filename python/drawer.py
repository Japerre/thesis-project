import os
import sys
import json
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import ast
from sklearn.ensemble import GradientBoostingClassifier
from joblib import load
from fractions import Fraction
from pprint import pprint
from multiprocessing import Pool
from tqdm import tqdm


BETA = '\u03B2'

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
	global DATASET_NAME, K_LIST, B_LIST, L_LIST
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


def certainty_plots_balanced_sample_v2(fold_number, k, b):
	bsample_v1_certainty_path = OUTPUT_BASE_PATH/f'BSample/fold_{fold_number}/k{k}/b({b})/privacystats/certainty.csv'
	bsample_v2_certainty_path = OUTPUT_BASE_PATH/f'BSample_v2/fold_{fold_number}/k{k}/b({b})/privacystats/certainty.csv'

	bsample_data = pd.read_csv(bsample_v1_certainty_path, sep=';', decimal=',')
	bsample_v2_data = pd.read_csv(bsample_v2_certainty_path, sep=';', decimal=',')

	# Combine data into one DataFrame with a 'version' column
	bsample_data['version'] = 'bsample_v1'
	bsample_v2_data['version'] = 'bsample_v2'
	combined_data = pd.concat([bsample_data, bsample_v2_data])

	# Create the violin plot with seaborn
	sns.violinplot(x='version', y='0', data=combined_data, cut=0)
	plt.xlabel('')
	plt.ylabel('certainty')
	plt.title(f'stratified balanced sampling voor k={k}, {BETA}={b}')
	
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

def violin_plots(fold, k, metric, sample_strat):
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
	sns.violinplot(x='B', y='0', data=all_data, cut=0)  # Replace 'your_column_name' with the name of the column you want to plot

	plt.title(f'Violin Plot of {metric} by B Value for {sample_strat}, k={k}, fold={fold}')
	plt.xlabel('B Value')
	plt.ylabel(metric)  # Adjust labels as needed
	plt.xticks(rotation=45)  # In case B values are not nicely numbered or if there are many B values

	plt.tight_layout()
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

			
def grouped_bar_chart_big_image(
		sample_strats: list, 
		experiment_num,
		title=None,
		target_translation_dict=None, 
		plot_non_masked=True,
		plot_std=False, 
		ros=False,
		rus=False,
		ldiv=False
		):
	
	for sample_strat in sample_strats:
		ml_experiments_dir = OUTPUT_BASE_PATH/sample_strat/'ml_experiments'/f'experiment_{experiment_num}'
		output_dir = OUTPUT_BASE_PATH / 'ml_plots' / f'experiment_{experiment_num}' / sample_strat / 'grouped_bar_chart'
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		df, avg_strats = classification_report_json_to_df(ml_experiments_dir, ldiv=ldiv)

		metrics = ['precision', 'recall', 'f1-score']
		fraction_b_values = [Fraction(value).limit_denominator() for value in B_LIST]

		# Create a single figure for all subplots
		fig, axes = plt.subplots(len(avg_strats), len(metrics), figsize=(15, 10))
		fig.suptitle(f'{DATASET_NAME} {sample_strat}', fontsize=16) if title is None else fig.suptitle(title, fontsize=16)
		
		for i, avg_strat in enumerate(avg_strats):
			if avg_strat not in ['macro avg', 'weighted avg'] and target_translation_dict is not None:
				target_name = target_translation_dict[avg_strat]
			else:
				target_name = avg_strat
			for j, metric in enumerate(metrics):
				mean_dict = {}
				std_dict = {}
				if ldiv:
					for k in K_LIST:
						k_rows = df.query(f'k == {k}').sort_values(by='l', ascending=True)
						mean_dict[k] = k_rows[f'{avg_strat}.mean_{metric}'].tolist()
						std_dict[k] = k_rows[f'{avg_strat}.std_{metric}'].tolist()
					x = np.arange(len(L_LIST))
				
				if not ldiv:
					for k in K_LIST:
						k_rows = df.query(f'k == {k}').sort_values(by='b', ascending=False)
						mean_dict[k] = k_rows[f'{avg_strat}.mean_{metric}'].tolist()
						std_dict[k] = k_rows[f'{avg_strat}.std_{metric}'].tolist()
					x = np.arange(len(B_LIST))
				
				width = 0.13
				multiplier = 0

				for key, value in mean_dict.items():
					offset = width * multiplier
					std_values = std_dict[key]
					if plot_std:
						rects = axes[i, j].bar(x + offset, value, width, label=f'k={key}', yerr=std_values)
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


def classification_report_json_to_df(ml_experiments_dir: Path, ldiv=False):
	df = pd.DataFrame()
	for k_dir in [d for d in ml_experiments_dir.iterdir() if d.name.startswith('k')]:
		k_val = int(k_dir.name[1:])
		if not ldiv:
			for b_dir in k_dir.iterdir():
				b_val = float(b_dir.name[2:-1])
				with open(b_dir/'classification_report.json', 'r') as json_file:
					report = json.load(json_file)
					flat_json = pd.json_normalize(report)
					flat_json['k'] = k_val
					flat_json['b'] = b_val
				df = pd.concat([df, flat_json], ignore_index=True)
		if ldiv:
			for l_dir in k_dir.iterdir():
				l_val = float(l_dir.name[1:])
				with open(l_dir/'classification_report.json', 'r') as json_file:
					report = json.load(json_file)
					flat_json = pd.json_normalize(report)
					flat_json['k'] = k_val
					flat_json['l'] = l_val
				df = pd.concat([df, flat_json], ignore_index=True)
	avg_strats = [key for key in report.keys() if key not in ['accuracy', 'train_record_counts']]
	return (df, avg_strats)


def cmp_balancing_pre_post(
		title=None,
		target_translation_dict=None, 
		):
	
	BSample_dir = OUTPUT_BASE_PATH/'BSample'/'ml_experiments'/'experiment_1'
	SSample_dir = OUTPUT_BASE_PATH/'SSample'/'ml_experiments'/'experiment_3'

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
	plt.savefig(output_dir / 'diff_pre_post_balancing.png', bbox_inches='tight')
	plt.close()


def test():
	df = pd.read_csv(r'C:\Users\tibol\Desktop\FIIW Tibo Laperre\fase 5 - thesis\thesis-projectV3\data\results\ACSIncome_USA_2018_binned_imbalanced_1664500\BSample\fold_0\k5\B(0.5)\B(0.5)_sample.csv', sep=';', decimal=',')
	# print(df)

NUM_PROCESSES = None

if __name__ == '__main__':
	NUM_PROCESSES = int(sys.argv[1])
	config_path = 'config/cmc.ini'
	# config_path = 'config/nursery.ini'
	# config_path = 'config\ACSIncome_USA_2018_binned_imbalanced_16645_acc_metric.ini'
	# config_path = 'config/ACSIncome_USA_2018_binned_imbalanced_16645.ini'
	# config_path = 'config/ACSIncome_USA_2018_binned_imbalanced_1664500.ini'
	read_config(config_path)
	# test()
	# violin_plots(0, 10, 'certainty', 'BSAMPLE')
	# privacy_plots_worker(['BSample'], certainty=True, journalist_risk=True)
	# certainty_plots_balanced_sample_v2(0, 5, 0.125)
	# grouped_bar_chart_big_image(['lDiv'], 1, ldiv=True, plot_std=True)
	# grouped_bar_chart_big_image(['SSAMPLE', 'BSAMPLE'], 1, target_translation_dict=ASCIncome_target_names(), plot_std=True)
	grouped_bar_chart_big_image(['lDiv'], 1, ldiv=True, target_translation_dict=cmc_target_names(), plot_std=True)
	# grouped_bar_chart_big_image(['SSAMPLE','BSAMPLE'], 1, plot_std=True)
	# grouped_bar_chart_big_image(['SSAMPLE'], 3, target_translation_dict=ASCIncome_target_names(), rus=True, title='ASCIncome RUS balancing after SSample', plot_std=True)
	# grouped_bar_chart_big_image(
	# 	sample_strats=['SSAMPLE', 'BSAMPLE', 'RSAMPLE'], 
	# 	experiment_num=1,
	# 	target_translation_dict=ASCIncome_target_names(), 
	# 	# rus=True,
	# 	# title="ASCIncome RUS balancing after SSample"
	# 	)
	# cmp_balancing_pre_post("ASCIncome BSample vs SSample with RUS in ML pipeline", ASCIncome_target_names())
	# grouped_bar_chart_big_image(['SSAMPLE'], 3, target_translation_dict=ASCIncome_target_names(), rus=True, title='ASCIncome RUS balancing after SSample')
	# grouped_bar_chart_big_image(['SSAMPLE'], 3, target_translation_dict=cmc_target_names(), ros=True, title='CMC ROS balancing after SSample', plot_std=True)
	

	
	