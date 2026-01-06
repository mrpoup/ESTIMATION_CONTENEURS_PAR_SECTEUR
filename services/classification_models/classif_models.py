import numpy as np
from sklearn.linear_model import LogisticRegression

class ModelLGBMClassifier:
    def __init__(self, random_state=42, **kwargs):
        from lightgbm import LGBMClassifier
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = LGBMClassifier(
            objective="multiclass",
            random_state=random_state,
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            **kwargs
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class ModelLogRegMultinomial:
    def __init__(self, random_state=42, max_iter=2000, **kwargs):
        self.model = LogisticRegression(
            solver="lbfgs",
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
