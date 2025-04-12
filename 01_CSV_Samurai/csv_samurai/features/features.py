from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class CustomLogTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to apply log transformation to specified columns.
    """
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[col] = X[col].apply(lambda x: np.log1p(x) if np.notnull(x) else x)
        return X
    
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.columns]