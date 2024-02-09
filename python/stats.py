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
	

def generelization_stats(k_list, num_folds, k_anon_base_path, qid_list):
	header = ['k','n']+qid_list+['suppressed_cnt', 'std_suppressed', 'sample_size', 'std_sample_size', 'EQ_cnt', 'std_EQ', 'avg_EQ_size', 'std_avg_EQ_size']
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
				record += [group_df[col].mean(), group_df[col].std()]
			records.append(record)
	
	df = pd.DataFrame(records, columns=header)
	df.fillna(0, inplace=True)
	generelization_stats_path = k_anon_base_path/'stats'/'generelization_stats.csv'
	generelization_stats_path.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(generelization_stats_path, sep=';', index=False)

	num_cols = df.select_dtypes(include='number').columns
	df[num_cols] = df[num_cols].astype(int)
	
	ax, table, col_widths = plot_table(np.vstack((df.columns.values, df.values.astype(str))))
	plt.savefig(generelization_stats_path.parent/'generelization_stats.png', dpi=500)


def sample_stats(strats_to_run, k_list, b_list, NUM_FOLDS, OUTPUT_BASE_PATH, QID_LIST, target):
	target_count_check = 0
	for strat in strats_to_run:
		result_data = []
		for k in k_list:
			for b in b_list:
				target_counts_list = []
				for fold in range(NUM_FOLDS):
					sample_path = OUTPUT_BASE_PATH/strat/f'fold_{fold}'/f'k{k}'/f'B({b})'/f'B({b})_sample.csv'
					sample_df = pd.read_csv(sample_path, delimiter=';')
					sample_df = drop_suppressed(sample_df, QID_LIST)
					target_counts = sample_df[target].value_counts()
					
					if target_count_check != 0 and target_count_check != len(target_counts):
						raise ValueError(f"strat: {strat}, fold: {fold}, k: {k}, b: {b} ==> does not have all classes in train set")
					else:
						target_count_check = len(target_counts)
					
					target_counts_list.append(target_counts)
				
				target_counts_df = pd.concat(target_counts_list, axis=1)
				mean_counts = target_counts_df.mean(axis=1)
				std_counts = target_counts_df.std(axis=1)
				
				result_data.append({
					'Strat': strat, 'k': k, 'b': b, 
					**{f"{target} Mean": mean for target, mean in zip(mean_counts.index, mean_counts)}, 
					**{f"{target} Std": std for target, std in zip(mean_counts.index, std_counts)}
					})
	

		result_df = pd.DataFrame(result_data)
		sample_stats_path = OUTPUT_BASE_PATH/strat/'stats'/'sample_stats.csv'
		sample_stats_path.parent.mkdir(parents=True, exist_ok=True)
		result_df.to_csv(sample_stats_path, sep=';',index=False)
	
					




