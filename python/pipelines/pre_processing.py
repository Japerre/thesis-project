from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

class StarredColTransformer(BaseEstimator, TransformerMixin):
	
	def __init__(self):
		pass

	def fit(self, X, y=None):
		return self

	def transform(self, X, y=None):
		X = X.copy()
		for column in X:
			unique = X[column].unique()
			if len(unique)==1 and '*' in unique:
				X[column] = -1
		
		# print(X)
		return X

class RangeTransformer(BaseEstimator, TransformerMixin):
	@staticmethod
	def transform_column(value):
		if isinstance(value, str) and '*' not in value:
			tmp = value[1:-1].split(', ')
			min, max = int(tmp[0]), int(tmp[1])
			return (min + max) / 2
		else:
			return value

	def __init__(self):
		pass

	def fit(self, X, y=None):
		return self

	def transform(self, X, y=None):
		X = X.copy()
		for column in X:
			unique = X[column].unique()
			if len(unique) == 1 and '*' in unique:
				# column is suppressed
				X[column] = 0
			elif X[column].dtype == 'O':
				X[column] = X[column].transform(self.transform_column)
		# print(X.shape)
		# print(X)
		return X
	

def filter_suppressed(X, y, qid):
	if qid is not None:
		X = X.copy()
		y = y.copy()
		groups = X.groupby(qid)
		key = tuple(['*'] * len(qid))
		if key in groups.groups:
			suppressed_rows = groups.get_group(name=key).index
			X = X.drop(suppressed_rows)
			y = y.drop(suppressed_rows)
	return X.infer_objects(), y



	
