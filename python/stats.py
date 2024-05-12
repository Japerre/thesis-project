import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from helper.table_plotter import plot_table
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import json


def drop_suppressed(df, qid_list):
	try:
		suppressed_rows = df.groupby(qid_list).get_group(name=tuple(['*'] * len(qid_list))).index
		return df.drop(suppressed_rows, inplace=False).infer_objects()
	except KeyError:
		# there are no suppressed records
		return df
	
def format_mean_spread(mean, spread):
	return f"{mean:.1f} ± {spread:.1f}"  # Format mean and std with one decimal precision

def generalization_stats(k_list, num_folds, k_anon_base_path, output_base_path, qid_list):
	header = ['k','n']+qid_list+['suppressed_cnt', 'sample_size','EQ_cnt', 'avg_EQ_size']
	records = []
	num_cols = ['suppressed in sample', 'sample size', 'equivalence classes', 'average EQ size']
	for k in k_list:
		stats_df_combined = pd.DataFrame()
		for fold_num in range(num_folds):
			stats_file = k_anon_base_path/f"fold_{fold_num}"/f"k{k}"/"stats.json"
			with open(stats_file, 'r') as file:
				stats_data = json.load(file)
				stats_df = pd.json_normalize(stats_data)
				stats_df['node'] = stats_df['node'].apply(lambda x: str(x))
			stats_df_combined = pd.concat([stats_df_combined, stats_df], ignore_index=True)
		
		groups = stats_df_combined.groupby('node')
		for group_name, group_df in groups:
			gen_levels = group_name[1:-1].split(', ')
			gen_levels = [int(level) for level in gen_levels]
			record = [k, len(group_df)]+gen_levels
			for col in num_cols:
				mean = group_df[col].mean()
				std = group_df[col].std()
				record.append(format_mean_spread(mean, std))
			records.append(record)
	
	df = pd.DataFrame(records, columns=header)
	# df.fillna(0, inplace=True)
	generalization_stats_path = output_base_path/'stats'/'generalization_stats.csv'
	generalization_stats_path.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(generalization_stats_path, sep=';', index=False)


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


def draw_combined_eq_distributions(strats_to_run, k_list, b_list, num_folds, output_base_path, k_anon_base_path, qid_list):
	for strat in strats_to_run:
		for k in k_list:
			k_anon_path = k_anon_base_path / f'fold_0' / f'k{k}' / 'output_sample.csv'
			k_anon_df = pd.read_csv(k_anon_path, sep=';', decimal=',')
			k_anon_groups = k_anon_df.groupby(qid_list).size()
			bins = np.arange(k_anon_groups.min() - 0.5, k_anon_groups.max() + 1.5, 1)

			for b in b_list:
					output_path = output_base_path / 'stats' / strat / 'eq_distributions' / f'fold_0' / f'k{k}' / f'B({b})' / 'distribution.png'
					output_path.parent.mkdir(parents=True, exist_ok=True)
					sample_path = output_base_path / strat / f'fold_0' / f'k{k}' / f'B({b})' / f'B({b})_sample.csv'
					sample_df = pd.read_csv(sample_path, sep=';', decimal=',')
					sample_groups = sample_df.groupby(qid_list).size()
					common_eqs = k_anon_groups[k_anon_groups.index.isin(sample_groups.index)]
					plt.figure(figsize=(10, 6))
					plt.hist(k_anon_groups, bins=bins, alpha=0.3, label='K-Anon EQ Sizes', color='blue')
					plt.hist(common_eqs, bins=bins, alpha=0.7, label='Sample EQ Sizes', color='orange')
					plt.legend()
					plt.title(f'EQ Size Distribution for k={k}, b={b}, strat={strat}')
					plt.xlabel('EQ Size')
					plt.ylabel('Frequency')
					plt.grid(axis='y', alpha=0.75)
					plt.savefig(output_path)
					plt.close()

					

def eq_stats(strats_to_run, k_list, b_list, num_folds, k_anon_base_path, output_base_path, stats_base_path, qid_list):
	for strat in strats_to_run:
		output_path = stats_base_path / strat / 'eq_stats.csv'
		eq_stats_data = []

		for k in k_list:
			for b in b_list:
				remaining_eq_sizes_kAnon = []
				remaining_eq_sizes_sample = []

				for fold in range(num_folds):
					k_anon_path = k_anon_base_path / f'fold_{fold}' / f'k{k}' / 'output_sample.csv'
					sample_path = output_base_path / strat / f'fold_{fold}' / f'k{k}' / f'B({b})' / f'B({b})_sample.csv'
					
					k_anon_df = pd.read_csv(k_anon_path, sep=';', decimal=',')
					sample_df = pd.read_csv(sample_path, sep=';', decimal=',')

					k_anon_groups = k_anon_df.groupby(qid_list).size()
					sample_groups = sample_df.groupby(qid_list).size()

					common_eqs = k_anon_groups.index.intersection(sample_groups.index)

					if common_eqs.empty:
						avg_eq_size_kAnon = 0
						avg_eq_size_sample = 0
					else:
						avg_eq_size_kAnon = k_anon_groups.loc[common_eqs].mean()
						avg_eq_size_sample = sample_groups.loc[common_eqs].mean()

					remaining_eq_sizes_kAnon.append(avg_eq_size_kAnon)
					remaining_eq_sizes_sample.append(avg_eq_size_sample)

				# Aggregate results for all folds
				avg_eq_size_kAnon_aggregated = np.mean(remaining_eq_sizes_kAnon)
				avg_eq_size_kAnon_aggregated_std = np.std(remaining_eq_sizes_kAnon)
				avg_eq_size_sample_aggregated = np.mean(remaining_eq_sizes_sample)
				avg_eq_size_sample_aggregated_std = np.std(remaining_eq_sizes_sample)

				eq_stats_data.append({
					'Strat': strat,
					'k': k,
					'b': b,
					'Avg EQ Size kAnon': format_mean_spread(avg_eq_size_kAnon_aggregated, avg_eq_size_kAnon_aggregated_std),
					'Avg EQ Size Sample': format_mean_spread(avg_eq_size_sample_aggregated, avg_eq_size_sample_aggregated_std)
				})

		strat_results_df = pd.DataFrame(eq_stats_data)
		strat_results_df.to_csv(output_path, index=False)


def sample_stats(strats_to_run, k_list, b_list, num_folds, output_base_path, qid_list, target_col, targets):
	for strat in strats_to_run:
		missing_targets_data = []
		result_data = []
		for k in k_list:
			for b in b_list:
				target_counts_list = []
				min_counts = {target: np.inf for target in targets}
				max_counts = {target: -np.inf for target in targets}
				for fold in range(num_folds):
					sample_path = output_base_path/strat/f'fold_{fold}'/f'k{k}'/f'B({b})'/f'B({b})_sample.csv'
					sample_df = pd.read_csv(sample_path, delimiter=';')
					sample_df = drop_suppressed(sample_df, qid_list)
					target_counts = sample_df[target_col].value_counts()
					missing_targets = set(targets) - set(target_counts.index)
					for missing_target in missing_targets:
						min_counts[missing_target] = 0
						max_counts[missing_target] = max(max_counts[missing_target], 0)
						print(f"strat: {strat}, fold: {fold}, k: {k}, b: {b} ==> does not have {missing_target} in train set")
						missing_targets_data.append({"strat": strat, "fold": fold, "k": k, "b": b, "target": missing_target})

					for target, count in target_counts.items():
						min_counts[target] = min(min_counts[target], count)
						max_counts[target] = max(max_counts[target], count)
						
					target_counts_list.append(target_counts)

				target_counts_df = pd.concat(target_counts_list, axis=1)
				mean_counts = target_counts_df.mean(axis=1)
				std_counts = target_counts_df.std(axis=1)
				
				min_max_col = {f'{target} range': f'[{min_counts[target]}, {max_counts[target]}]' for target in min_counts.keys()}
				formatted_data = {f"{target}": format_mean_spread(mean, std) for target, mean, std in zip(mean_counts.index, mean_counts, std_counts)}
				result_data.append({'Strat': strat, 'k': k, 'b': b, **formatted_data, **min_max_col})

		result_df = pd.DataFrame(result_data)
		sample_stats_path = output_base_path/'stats'/f'{strat}'/'target_counts_stats.csv'
		sample_stats_path.parent.mkdir(parents=True, exist_ok=True)
		result_df.to_csv(sample_stats_path, sep=';',index=False)
		
		if missing_targets_data:
			missing_targets_df = pd.DataFrame(missing_targets_data)
			missing_targets_path = output_base_path/'stats'/f'{strat}'/'missing_targets.csv'
			missing_targets_df.to_csv(missing_targets_path, sep=';', index=False)


def eq_per_target(strats_to_run, k_list, b_list, num_folds, output_base_path, qid_list, target, targets):
	for strat in strats_to_run:
		result_data = []
		for k in k_list:
			for b in b_list:
				eq_counts_dict_list = []
				min_counts = {target: np.inf for target in targets}
				max_counts = {target: -np.inf for target in targets}
				for fold in range(num_folds):
					sample_path = output_base_path/strat/f'fold_{fold}'/f'k{k}'/f'B({b})'/f'B({b})_sample.csv'
					sample_df = pd.read_csv(sample_path, delimiter=';')
					sample_df = drop_suppressed(sample_df, qid_list)
					target_groups = sample_df.groupby(target)
					missing_targets = set(targets) - set(target_groups.groups.keys())
					for missing_target in missing_targets:
						min_counts[missing_target] = 0
						max_counts[missing_target] = max(max_counts[missing_target], 0)
						print(f"strat: {strat}, fold: {fold}, k: {k}, b: {b} ==> does not have {missing_target} in train set")

					eq_counts_dict = {}
					for group_name, target_group_df in target_groups:
						eq_count = target_group_df.groupby(qid_list).ngroups
						eq_counts_dict[group_name] = eq_count
						min_counts[group_name] = min(min_counts[group_name], eq_count)
						max_counts[group_name] = max(max_counts[group_name], eq_count)
					eq_counts_dict_list.append(eq_counts_dict)

				
				eq_counts_df = pd.DataFrame(eq_counts_dict_list)
				eq_counts_df = eq_counts_df.fillna(0)
				mean_counts = eq_counts_df.mean(axis=0)
				std_counts = eq_counts_df.std(axis=0)
				
				formatted_data = {f"{target}": format_mean_spread(mean, std) for target, mean, std in zip(mean_counts.index, mean_counts, std_counts)}
				min_max_col = {f'{target} range': f'[{min_counts[target]}, {max_counts[target]}]' for target in min_counts.keys()}
				result_data.append({'Strat': strat, 'k': k, 'b': b, **formatted_data, **min_max_col})
		result_df = pd.DataFrame(result_data)
		eq_per_target_stats_path = output_base_path/'stats'/f'{strat}'/'eq_counts_per_target.csv'
		eq_per_target_stats_path.parent.mkdir(parents=True, exist_ok=True)
		result_df.to_csv(eq_per_target_stats_path, sep=';',index=False)

def find_biggest_certainty(bsample_base_path, num_folds, k_list, b_list):
	max_cert = 0
	max_certainty_path = None
	for k in k_list:
		for b in b_list:
			for fold in range(num_folds):
				certainty_path = bsample_base_path/f'fold_{fold}'/f'k{k}'/f'B({b})'/'privacystats'/'certainty.csv'
				certainty_df = pd.read_csv(certainty_path, decimal=',', sep=';')
				local_max_cert = certainty_df['0'].max()
				if local_max_cert > max_cert:
					max_cert = local_max_cert
					max_certainty_path = certainty_path
	
	print(max_certainty_path)
	print(max_cert)

def find_biggest_procentual_certainty(bsample_base_path, num_folds, k_list, b_list):
	max_cert_procentual_diff = 0.0
	max_certainty_procentual_diff_path = None
	for k in k_list:
		for b in b_list:
			for fold in range(num_folds):
				certainty_path = bsample_base_path/f'fold_{fold}'/f'k{k}'/f'B({b})'/'privacystats'/'certainty.csv'
				certainty_df = pd.read_csv(certainty_path, decimal=',', sep=';')
				local_max_cert = certainty_df['0'].max()
				diff = (local_max_cert-b)/b
				print(diff)
				if diff > max_cert_procentual_diff:
					max_cert_procentual_diff = diff
					max_certainty_procentual_diff_path = certainty_path
	
	print(max_certainty_procentual_diff_path)
	print(max_cert_procentual_diff)


def print_distribution(params):
	sample_path, output_path, target = params
	df = pd.read_csv(sample_path, sep=';', decimal=',')
	sns.histplot(data=df, x=target, kde=False, color='purple')
	plt.xlabel(target)
	plt.ylabel('Count')
	plt.title(f'Distribution of {target}')
	sns.despine()
	plt.savefig(output_path)
	plt.close()

def print_distributions_worker(sample_base_path, num_folds, k_list, b_list, target, num_processes=4):
	jobs = []
	for fold in range(num_folds):
		for k in k_list:
			for b in b_list:
				sample_path = sample_base_path/f'fold_{fold}/k{k}/B({b})/B({b})_sample.csv'
				output_path = sample_base_path/f'fold_{fold}/k{k}/B({b})/distribution.png'
				jobs.append((sample_path, output_path, target))
	
	with Pool(processes=num_processes) as pool:
		result = list(tqdm(pool.imap(print_distribution, jobs),total=len(jobs)))


def print_fully_suppressed_samples_ldiv(ldiv_base_path, num_folds, k_list, l_list):
	fully_suppressed_samples = []
	for k in k_list:
		for l in l_list:
			for fold in range(num_folds):
				sample_stats_path = ldiv_base_path/f'fold_{fold}/k{k}/l{l}/stats.json'
				with open(sample_stats_path, 'r') as json_file:
					stats = json.load(json_file)
				if stats['sample size'] == 0:
					fully_suppressed_samples.append(f'fold: {fold} k: {k} l: {l}')
	
	for entry in fully_suppressed_samples:
		print(entry)

def check_for_fully_suppressed(params):
		sample_path, qid = params
		try:
				sample_df = pd.read_csv(sample_path, sep=';', decimal=',')
				not_fully_suppressed = ~(sample_df[qid] == '*').all(axis=1)
				if not not_fully_suppressed.any():
						# all records are suppressed
						print(sample_path)
						return sample_path
		except Exception as e:
				print(f"Error processing file {sample_path}: {e}")
		return None

def print_fully_suppressed_samples(sample_base_path, num_folds, k_list, b_list, qid, num_processes=8):
		jobs = []
		for fold in range(num_folds):
				for k in k_list:
						for b in b_list:
								sample_path = sample_base_path / f'fold_{fold}' / f'k{k}' / f'B({b})' / f'B({b})_sample.csv'
								jobs.append((sample_path, qid))
		
		# Use multiprocessing to process files and track progress with tqdm
		with Pool(processes=num_processes) as pool:
				results = list(tqdm(pool.imap(check_for_fully_suppressed, jobs), total=len(jobs), desc='Checking for fully suppressed samples'))

		# Filter out None results and print paths of files with fully suppressed samples
		fully_suppressed_samples = [result for result in results if result is not None]
		for path in fully_suppressed_samples:
				print(f"Fully suppressed sample found in: {path}")