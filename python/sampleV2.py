from pathlib import Path
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
from stats import drop_suppressed

class SampleV2:
	def __init__(self, sample_folder_base_path, k_anon_base_path, num_folds, k_list, b_list, qid, tolerance_percentage, num_processes):
		self.sample_folder_base_path = sample_folder_base_path
		self.kanon_base_path = k_anon_base_path
		self.num_folds = num_folds
		self.k_list = k_list
		self.b_list = b_list
		self.qid = qid
		self.tolerance_percentage = tolerance_percentage
		self.num_processes = num_processes
		self.sample_v2_base_path = Path(self.sample_folder_base_path.parent)/f'{self.sample_folder_base_path.name}_V2'
		self.sample_v2_base_path.mkdir(exist_ok=False)

	@staticmethod
	def _certainty(population_df: pd.DataFrame, sample_df: pd.DataFrame, qid: list, output:bool) -> pd.Series:
		population_eq = population_df.groupby(qid).size()
		sample_eq = sample_df.groupby(qid).size()
		if output:
			result = sample_eq.div(population_eq).fillna(0).reset_index(level=qid)
			result2 = sample_df.merge(result, on=qid, how='left')
			return result2[0]
		else:
			result = sample_eq.div(population_eq).fillna(0)
			return result

	@staticmethod
	def _take_sample(group, certainty, max_cert):
		if certainty<max_cert:
			return group
		else:
			amount = int((len(group)/certainty)*max_cert)
			return group.sample(amount, ignore_index=True).reset_index(drop=True)

	def _filter_df(self, params):
		sample_path, population_path, fold, k, b, qid, tolerance_percentage, sample_v2_base_path = params
		max_cert = b * (1 + tolerance_percentage)
		population_df = pd.read_csv(population_path, sep=';', decimal=',')
		population_df = drop_suppressed(population_df, self.qid)
		sample_df = pd.read_csv(sample_path, sep=';', decimal=',')
		sample_df = drop_suppressed(sample_df, self.qid)
		certainty_df = self._certainty(population_df, sample_df, qid, False)
		sample_df_filtered = sample_df.groupby(qid).apply(
					lambda group: self._take_sample(group, certainty_df[group.name], max_cert)).reset_index(drop=True)
		sample_df_filtered = sample_df_filtered.sample(frac=1).reset_index(drop=True)
		certainty_df_filtered = self._certainty(population_df, sample_df_filtered, qid, True)

		certainty_filtered_path = Path(sample_v2_base_path)/f'fold_{fold}/k{k}/B({b})/privacystats/certainty.csv'
		sample_filtered_path = Path(sample_v2_base_path)/f'fold_{fold}/k{k}/B({b})/B({b})_sample.csv'
		certainty_filtered_path.parent.mkdir(parents=True, exist_ok=True)
		sample_filtered_path.parent.mkdir(parents=True, exist_ok=True)
		certainty_df_filtered.to_csv(certainty_filtered_path, sep=';', decimal=',', index=False)
		sample_df_filtered.to_csv(sample_filtered_path, sep=';', decimal=',', index=False)

	def run(self):
		jobs = []
		for fold in range(self.num_folds):
			for k in self.k_list:
				population_path = Path(self.kanon_base_path)/f'fold_{fold}/k{k}/output_sample.csv'
				for b in self.b_list:
					sample_path = Path(self.sample_folder_base_path)/f'fold_{fold}/k{k}/B({b})/B({b})_sample.csv'
					jobs.append((sample_path, population_path, fold, k, b, self.qid, self.tolerance_percentage, self.sample_v2_base_path))
		
		with Pool(processes=self.num_processes) as pool:
			list(tqdm(pool.imap(self._filter_df, jobs), total=len(jobs), desc='sample_v2'))

