from imblearn.under_sampling import RandomUnderSampler
from imblearn import FunctionSampler
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from imblearn.pipeline import Pipeline
from .pre_processing import RangeTransformer, filter_suppressed


def get_non_masked_pipe(cat, verbose):
	return Pipeline([
		('preprocessing', ColumnTransformer(
			transformers=[("cat", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat),],
			remainder='passthrough',
			verbose=verbose)),
		('model', GradientBoostingClassifier(verbose=verbose, loss='log_loss'))
	], verbose=verbose)

def experiment_3_pipeline(verbose: bool, qid_list, cat):
	#pipeline to compare balancing on SSample (balancing in ML pipeline) vs BSample (balancing in anon pipeline)
	pipe = Pipeline([
			('suppressed_filter', FunctionSampler(func=filter_suppressed, kw_args={'qid': qid_list}, validate=False)),
			('preprocessing', ColumnTransformer(
				transformers=[
					("cat", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat)],
				remainder='passthrough',
				verbose=verbose)),
				('model', GradientBoostingClassifier(verbose=verbose, loss='log_loss'))
		], verbose=verbose)
	
	non_masked_balanced_RUS_pipe = Pipeline([
		('sampler', RandomUnderSampler()),
		('preprocessing', ColumnTransformer(
			transformers=[("cat", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat),],
			remainder='passthrough',
			verbose=verbose)),
		('model', GradientBoostingClassifier(verbose=verbose, loss='log_loss'))
	], verbose=verbose)

	non_masked_pipe = get_non_masked_pipe(cat, verbose)

	return pipe,non_masked_pipe,non_masked_balanced_RUS_pipe


def experiment_1_pipeline(verbose: bool, qid_list, cat):

	pipe = Pipeline([
		('suppressed filter', FunctionSampler(func=filter_suppressed, kw_args={'qid': qid_list}, validate=False)),
		('preprocessing', ColumnTransformer(
			transformers=[("cat", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat),],
			remainder='passthrough',
			verbose=verbose)),
			('model', GradientBoostingClassifier(verbose=verbose, loss='log_loss'))
	], verbose=verbose)

	non_masked_pipe = get_non_masked_pipe(cat, verbose)
	
	return pipe, non_masked_pipe