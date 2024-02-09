from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn import FunctionSampler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from imblearn.pipeline import Pipeline
from .pre_processing import RangeTransformer, filter_suppressed, StarredColTransformer


def experiment_3_pipeline(verbose: bool, qid_list):
	numerical_pipe = Pipeline([('range_to_mean', RangeTransformer()), ('scaler', StandardScaler())])

	num = ['WIFEAGE', 'WIFEEDU', 'HUSBEDU', 'CHLD']
	possibly_starred = list(set(qid_list) - set(num))
	# print(possibly_starred)

	pipe = Pipeline([
		('suppressed filter', FunctionSampler(func=filter_suppressed, kw_args={'qid': qid_list}, validate=False)),
		('over sampler', RandomOverSampler()),
		('preprocessing', ColumnTransformer(
			transformers=[
				("num", numerical_pipe, ['WIFEAGE', 'WIFEEDU', 'HUSBEDU', 'CHLD']),
				("star", StarredColTransformer(), possibly_starred)
				],
			remainder='passthrough',
			verbose=verbose)),
		('model', GradientBoostingClassifier(verbose=verbose, loss='log_loss'))
	], verbose=verbose)

	non_masked_balanced_ROS_pipe = Pipeline([
		('over sampler', RandomOverSampler()),
		('model', GradientBoostingClassifier(verbose=verbose, loss='log_loss'))
	], verbose=verbose)

	non_masked_pipe = Pipeline([
		('model', GradientBoostingClassifier(verbose=verbose, loss='log_loss'))
	], verbose=verbose)
	
	return pipe, non_masked_pipe, non_masked_balanced_ROS_pipe



def experiment_2_pipeline(verbose: bool, qid_list):
		return



def experiment_1_pipeline(verbose: bool, qid_list):
	numerical_pipe = Pipeline([('range_to_mean', RangeTransformer()), ('scaler', StandardScaler())])

	num = ['WIFEAGE', 'WIFEEDU', 'HUSBEDU', 'CHLD']
	possibly_starred = list(set(qid_list) - set(num))
	# print(possibly_starred)

	pipe = Pipeline([
		('suppressed filter', FunctionSampler(func=filter_suppressed, kw_args={'qid': qid_list}, validate=False)),
		('preprocessing', ColumnTransformer(
			transformers=[
				("num", numerical_pipe, ['WIFEAGE', 'WIFEEDU', 'HUSBEDU', 'CHLD']),
				("star", StarredColTransformer(), possibly_starred)
				],
			remainder='passthrough',
			verbose=verbose)),
		('model', GradientBoostingClassifier(verbose=verbose, loss='log_loss'))
	], verbose=verbose)

	non_generalized_pipe = Pipeline([
		# ('preprocessing', ColumnTransformer(
		# 	# transformers=[("cat", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat),],
		# 	remainder='passthrough',
		# 	verbose=verbose)),
		('model', GradientBoostingClassifier(verbose=verbose, loss='log_loss'))
	], verbose=verbose)
	
	return pipe, non_generalized_pipe