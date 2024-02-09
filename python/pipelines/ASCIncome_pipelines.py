from imblearn.under_sampling import RandomUnderSampler
from imblearn import FunctionSampler
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from imblearn.pipeline import Pipeline
from .pre_processing import RangeTransformer, filter_suppressed

def experiment_4_pipeline(verbose: bool, qid_list, cat):
	

	return

def experiment_3_pipeline(verbose: bool, qid_list, cat):
	#pipeline to compare balancing on SSample (balancing in ML pipeline) vs BSample (balancing in anon pipeline)
	numerical_pipe = Pipeline([('range_to_mean', RangeTransformer()), ('scaler', StandardScaler())])
	pipe = Pipeline([
			('suppressed_filter', FunctionSampler(func=filter_suppressed, kw_args={'qid': qid_list}, validate=False)),
			('sampler', RandomUnderSampler()),
			('preprocessing', ColumnTransformer(
				transformers=[
					("cat", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat),
					("num", numerical_pipe, ['AGEP'])],
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

	non_masked_pipe = Pipeline([
		('preprocessing', ColumnTransformer(
			transformers=[("cat", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat),],
			remainder='passthrough',
			verbose=verbose)),
		('model', GradientBoostingClassifier(verbose=verbose, loss='log_loss'))
	], verbose=verbose)
	
	return pipe,non_masked_pipe,non_masked_balanced_RUS_pipe


def experiment_1_pipeline(verbose: bool, qid_list, cat):
	numerical_pipe = Pipeline([('range_to_mean', RangeTransformer()), ('scaler', StandardScaler())])

	pipe = Pipeline([
		('suppressed filter', FunctionSampler(func=filter_suppressed, kw_args={'qid': qid_list}, validate=False)),
		('preprocessing', ColumnTransformer(
			transformers=[
				("cat", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat),
				("num", numerical_pipe, ['AGEP'])],
			remainder='passthrough',
			verbose=verbose)),
			('model', GradientBoostingClassifier(verbose=verbose, loss='log_loss'))
	], verbose=verbose)

	non_generalized_pipe = Pipeline([
		('preprocessing', ColumnTransformer(
			transformers=[("cat", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat),],
			remainder='passthrough',
			verbose=verbose)),
		('model', GradientBoostingClassifier(verbose=verbose, loss='log_loss'))
	], verbose=verbose)
	
	return pipe, non_generalized_pipe