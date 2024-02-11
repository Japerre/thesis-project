import pandas as pd
import matplotlib.pyplot as plt
from helper.table_plotter import plot_table
import numpy as np

def drop_suppressed(df, qid_list):
	try:
		suppressed_rows = df.groupby(qid_list).get_group(name=tuple(['*'] * len(qid_list))).index
		return df.drop(suppressed_rows, inplace=False).infer_objects()
	except KeyError:
		# there are no suppressed records
		return df
	
def format_mean_std(mean, std):
	return f"{mean:.1f} ± {std:.1f}"  # Format mean and std with one decimal precision


def generelization_stats(k_list, num_folds, k_anon_base_path, output_base_path, qid_list):
	# header = ['k','n']+qid_list+['suppressed_cnt', 'std_suppressed', 'sample_size', 'std_sample_size', 'EQ_cnt', 'std_EQ', 'avg_EQ_size', 'std_avg_EQ_size']
	header = ['k','n']+qid_list+['suppressed_cnt', 'sample_size','EQ_cnt', 'avg_EQ_size']
	records = []
	num_cols = ['suppressed in sample', 'sample size', 'equivalence classes', 'average EQ size']
	for k in k_list:
		stats_df_combined = pd.DataFrame()
		for fold_num in range(num_folds):
			stats_file = k_anon_base_path/f"fold_{fold_num}"/f"k{k}"/"stats.csv"
			stats_df = pd.read_csv(stats_file, sep=';', decimal='.')
			stats_df_combined = pd.concat([stats_df_combined, stats_df], ignore_index=True)
		
		groups = stats_df_combined.groupby('node')
		for group_name, group_df in groups:
			gen_levels = group_name[1:-1].split(', ')
			gen_levels = [int(level) for level in gen_levels]
			record = [k, len(group_df)]+gen_levels
			for col in num_cols:
				mean = group_df[col].mean()
				std = group_df[col].std()
				record.append(format_mean_std(mean, std))
			records.append(record)
	
	df = pd.DataFrame(records, columns=header)
	# df.fillna(0, inplace=True)
	generelization_stats_path = output_base_path/'stats'/'generelization_stats.csv'
	generelization_stats_path.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(generelization_stats_path, sep=';', index=False)


def sample_stats(strats_to_run, k_list, b_list, num_folds, output_base_path, qid_list, target):
	target_count_check = 0
	for strat in strats_to_run:
		result_data = []
		for k in k_list:
			for b in b_list:
				target_counts_list = []
				for fold in range(num_folds):
					sample_path = output_base_path/strat/f'fold_{fold}'/f'k{k}'/f'B({b})'/f'B({b})_sample.csv'
					sample_df = pd.read_csv(sample_path, delimiter=';')
					sample_df = drop_suppressed(sample_df, qid_list)
					target_counts = sample_df[target].value_counts()
					
					if target_count_check != 0 and target_count_check != len(target_counts):
						print(f"strat: {strat}, fold: {fold}, k: {k}, b: {b} ==> does not have all classes in train set")
						# raise ValueError(f"strat: {strat}, fold: {fold}, k: {k}, b: {b} ==> does not have all classes in train set")
					else:
						target_count_check = len(target_counts)
					
					target_counts_list.append(target_counts)
				
				target_counts_df = pd.concat(target_counts_list, axis=1)
				mean_counts = target_counts_df.mean(axis=1)
				std_counts = target_counts_df.std(axis=1)
				
				formatted_data = {f"{target}": format_mean_std(mean, std) for target, mean, std in zip(mean_counts.index, mean_counts, std_counts)}

				result_data.append({'Strat': strat, 'k': k, 'b': b, **formatted_data})

		result_df = pd.DataFrame(result_data)
		sample_stats_path = output_base_path/'stats'/f'{strat}'/'target_counts_stats.csv'
		sample_stats_path.parent.mkdir(parents=True, exist_ok=True)
		result_df.to_csv(sample_stats_path, sep=';',index=False)
	
					
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
						# if eq_count == 0:
						# 	print(f"strat: {strat}, fold: {fold}, k: {k}, b: {b} ==> does not have all classes in train set")
						eq_counts_dict[group_name] = eq_count
						min_counts[group_name] = min(min_counts[group_name], eq_count)
						max_counts[group_name] = max(max_counts[group_name], eq_count)
					eq_counts_dict_list.append(eq_counts_dict)

				
				eq_counts_df = pd.DataFrame(eq_counts_dict_list)
				eq_counts_df = eq_counts_df.fillna(0)
				mean_counts = eq_counts_df.mean(axis=0)
				std_counts = eq_counts_df.std(axis=0)
				
				formatted_data = {f"{target}": format_mean_std(mean, std) for target, mean, std in zip(mean_counts.index, mean_counts, std_counts)}
				min_max_col = {f'{target} range': f'[{min_counts[target]}, {max_counts[target]}]' for target in min_counts.keys()}
				result_data.append({'Strat': strat, 'k': k, 'b': b, **formatted_data, **min_max_col})
		result_df = pd.DataFrame(result_data)
		eq_per_target_stats_path = output_base_path/'stats'/f'{strat}'/'eq_counts_per_target.csv'
		eq_per_target_stats_path.parent.mkdir(parents=True, exist_ok=True)
		result_df.to_csv(eq_per_target_stats_path, sep=';',index=False)
