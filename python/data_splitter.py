import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.model_selection import KFold
import os

def main():
	if len(sys.argv) < 3:
		print("Usage: python train_test_split.py <input_dataset_file> <train_dataset_percentage>")
		sys.exit(99)

	input_dataset_file = Path(sys.argv[1])
	cross_validate = bool(sys.argv[2])

	FOLD_DIR = input_dataset_file.parent / 'folds'
	df = pd.read_csv(input_dataset_file, delimiter=';')
   
	if cross_validate:
		kf = KFold(n_splits=10, shuffle=True)
		for fold_number, (train_indices, test_indices) in enumerate(kf.split(df)):
			train_set = df.iloc[train_indices]
			test_set = df.iloc[test_indices]
			output_folder = FOLD_DIR/f'fold_{fold_number}'
			os.makedirs(output_folder)
			train_set.to_csv(output_folder/'train.csv', index=False, sep=';')
			test_set.to_csv(output_folder/'test.csv', index=False, sep=';')

if __name__ == "__main__":
	main()
