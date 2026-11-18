"""scikit-learn transformer exposing the interpretable cue features."""
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from .lexicon import extract_features, FEATURE_NAMES


class RuleFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for t in X:
            f = extract_features(t)
            rows.append([f[n] for n in FEATURE_NAMES])
        return sparse.csr_matrix(np.asarray(rows, dtype=float).reshape(len(rows), len(FEATURE_NAMES)))
