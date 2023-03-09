from typing import Callable
import numpy as np
import sys
sys.path.append('..')
from Aula1.Dataset import Dataset
from f_classification import FClassification

class SelectKBest:
    """
    Selecionar features conforme k scores mais altos.
    O ranking das features é feito fazendo a computação dos scores de cada feature usando uma `scoring function`.
        - f_classification: ANOVA F-value between label/feature for classification tasks.
        - f_regression: F-value obtained from F-value of r's pearson correlation coefficients for regression tasks.
    """

    def __init__(self, score_func: Callable = FClassification.fit_transform, k: int = 10):
        """
        Selecionar features conforme o k scores mais altos..
        """

        self.k = k
        self.score_func = score_func
        self.F = None
        self.p = None
    
    def fit(self, dataset: Dataset) -> 'SelectKBest':
        """
        Faz o fit SelectKBest
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Transformar o dataset selecionando as k features com maior score.
        """
        idxs = np.argsort(self.F)[-self.k:]
        features = np.array(dataset.features)[idxs]
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)
    
    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fit e de seguida transform.
        """
        self.fit(dataset)
        return self.transform(dataset)
    