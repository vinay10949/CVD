import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from feature_engine.discretisers import DecisionTreeDiscretiser
from feature_engine.categorical_encoders import OneHotCategoricalEncoder
import category_encoders as ce


def calculateBloodPressureLevel(data):
    if (data["ap_hi"] < 120) and (data["ap_lo"] < 80):
        return "Normal"
    if (data["ap_hi"] >= 120 and data["ap_hi"] <= 129) and (data["ap_lo"] < 80):
        return "Elevated"
    if (data["ap_hi"] >= 130 and data["ap_hi"] <= 139) | (
        data["ap_lo"] >= 80 and data["ap_lo"] <= 89
    ):
        return "Stage1HyperTension"
    if (data["ap_hi"] >= 140) | (data["ap_lo"] >= 90):
        return "Stage2HyperTension"
    if (data["ap_hi"] >= 180) | (data["ap_lo"] >= 120):
        return "HypertensiveCrisis"


def BMI(data):
    return data["weight"] / (data["height"] / 100) ** 2


class DiscretizeVariable(BaseEstimator, TransformerMixin):
    """Drop Duplicates from the data."""

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
        self.treeDisc = DecisionTreeDiscretiser(
            cv=10,
            scoring="roc_auc",
            variables=self.variables,
            regression=False,
            param_grid={"max_depth": [1, 2], "min_samples_leaf": [10, 4, 6]},
        )

    def fit(self, X, y=None):
        # We need this step for sklearn pipline
        self.treeDisc.fit(X, y)
        return self

    def transform(self, X) -> pd.DataFrame:
        X = X.copy()
        X = self.treeDisc.transform(X)
        return X


class DiscretizeBMI(BaseEstimator, TransformerMixin):
    """DiscretizeBMI Estimator."""

    def __init__(self,):
        return None

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X) -> pd.DataFrame:
        X = X.copy()
        # bucket boundaries
        buckets = [0, 18.5, 24.9, 29.9, 1000]
        # bucket labels
        labels = ["Underweight", "Healthy", "Overweight", "Obese"]
        # discretisation
        X["bmi_category"] = pd.cut(
            X["bmi"], bins=buckets, labels=labels, include_lowest=True
        )
        X["bmi_category"] = X["bmi_category"].astype("object")
        return X


class CalculateBloodPressureLevel(BaseEstimator, TransformerMixin):
    """Calculate Blood Pressure Level  ."""

    def __init__(self,):
        return None

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X) -> pd.DataFrame:
        X = X.copy()
        X["blood_pressure_level"] = X.apply(calculateBloodPressureLevel, axis=1)
        X["blood_pressure_level"] = X["blood_pressure_level"].astype("object")
        return X


class OHEEncoder(BaseEstimator, TransformerMixin):
    """OHE Encoder categorical encoder"""

    def __init__(self, variables=None):
        self.variables = variables
        self.ohe_enc = OneHotCategoricalEncoder(
            variables=self.variables,  # we can select which variables to encode
            drop_last=True,
        )  # to return k-1, false to return k

    def fit(self, X, y=None):
        # persist frequent labels in dictionary
        for feature in X[self.variables]:
            X[feature] = X[feature].astype("object")
        self.ohe_enc.fit(X)
        return self

    def transform(self, X):
        X = X.copy()
        X = self.ohe_enc.transform(X)
        return X


class JamesStienEncoder(BaseEstimator, TransformerMixin):
    """String to numbers categorical encoder."""

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
        self.encoder = ce.JamesSteinEncoder(cols=self.variables)

    def fit(self, X, y):
        self.encoder.fit(X, y)
        return self

    def transform(self, X):
        # encode labels
        X = self.encoder.transform(X)
        return X


class DropUnecessaryFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, variables_to_drop=None):
        self.variables = variables_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        X = X.drop(self.variables, axis=1)
        return X
