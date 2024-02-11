import os
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
	DATASET_NAME = cfg['VARIABLES']['dataset_name']
	K_LIST = [5, 10, 20, 40, 80, 160]
	B_LIST = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
	L_LIST = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25]

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

def save_privacy_plots():
	privacy_folder_paths = []
	for foldername, subfolders, filenames in os.walk(OUTPUT_BASE_PATH):
		if(Path(foldername).name == 'privacystats'):
			privacy_folder_paths.append(foldername)
	
	for folder in privacy_folder_paths:
		# Define the paths
		certainty_path = os.path.join(folder, "certainty.csv")
		journalist_risk_path = os.path.join(folder, "journalistRisk.csv")
		certainty_plot_path = os.path.join(folder, "certainty.png")
		journalist_risk_plot_path = os.path.join(folder, "journalistRisk.png")

		data = pd.read_csv(certainty_path, decimal=',', sep=';')
		sns.violinplot(y=data['0'], cut=0)  
		plt.savefig(certainty_plot_path, bbox_inches='tight', dpi=300)
		plt.close()
		data = pd.read_csv(journalist_risk_path, decimal=',', sep=';')
		sns.violinplot(y=data['0'], cut=0)  
		plt.savefig(journalist_risk_plot_path, bbox_inches='tight', dpi=300)
		plt.close()
	


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


if __name__ == '__main__':
	# config_path = 'config/cmc.ini'
	config_path = 'config/nursery.ini'
	# config_path = 'config\ACSIncome_USA_2018_binned_imbalanced_16645_acc_metric.ini'
	# config_path = 'config/ACSIncome_USA_2018_binned_imbalanced_16645.ini'
	# config_path = 'config/income_binned_USA_1664500.ini'
	read_config(config_path)
	# save_privacy_plots()
	# grouped_bar_chart_big_image(['lDiv'], 1, ldiv=True, plot_std=True)
	# grouped_bar_chart_big_image(['SSAMPLE', 'BSAMPLE'], 1, target_translation_dict=cmc_target_names(), plot_std=True)
	# grouped_bar_chart_big_image(['SSAMPLE', 'BSAMPLE'], 1, target_translation_dict=cmc_target_names(), plot_std=True)
	grouped_bar_chart_big_image(['BSAMPLE'], 1, plot_std=True)
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
	

	
	